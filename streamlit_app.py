import streamlit as st
import weaviate
from weaviate.classes.init import Auth
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
import os
from typing import List, Dict, Tuple
from datetime import datetime
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import CommaSeparatedListOutputParser
import weaviate.classes as wvc
from pydantic import BaseModel, Field


class LegalDocumentEvaluation(BaseModel):
    """Evaluation of retrieved legal documents"""

    relevance_explanation: str = Field(
        description="Explanation of how the documents answer the query"
    )

    cited_articles: List[str] = Field(
        description="List of relevant article citations",
        min_length=1
    )

    missing_information: List[str] = Field(
        description="Information that would help provide a more accurate answer",
        default=[]
    )

    time_period_note: str = Field(
        description="Note about the time period relevance of the answer",
        default=""
    )


def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "conversation_memory" not in st.session_state:
        st.session_state.conversation_memory = ConversationBufferMemory(
            return_messages=True,
            memory_key="chat_history",
            input_key="input"
        )


def setup_weaviate_client():
    wcd_url = 'https://xh1j9trzu5cervreztxw.c0.europe-west3.gcp.weaviate.cloud'
    return weaviate.connect_to_weaviate_cloud(
        cluster_url=wcd_url,
        auth_credentials=Auth.api_key('uTTyayyrfwyn98zBq6ukAcIAVnEJjkBWMLac'),
        headers={"X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")}
    )


def extract_date_from_query(query: str, llm: ChatOpenAI) -> datetime:
    """Extract date from query using LLM"""
    template = """Extract the relevant date from the user's legal query.
    If no specific date is mentioned, determine if the query implies a historical or current context.

    User Query: {query}

    Rules:
    1. If a specific date is mentioned, return it in YYYY-MM-DD format
    2. If no date is mentioned but historical context is implied, estimate the relevant period
    3. If no temporal context is given, return today's date
    4. Always return a single date in YYYY-MM-DD format

    Return ONLY the date in YYYY-MM-DD format, nothing else."""

    prompt = ChatPromptTemplate.from_template(template)
    result = llm.invoke(prompt.format(query=query))

    try:
        return datetime.strptime(result.content.strip(), '%Y-%m-%d')
    except:
        return datetime.now()


def get_general_terms(client, query_date: datetime) -> List[Dict]:
    """Retrieve general terms articles valid for the given date"""
    try:
        collection = client.collections.get("Vmi_docs")

        # Create filter for general terms articles
        filter_general = wvc.query.Filter.by_property("article_number").contains_any(["1", "2"])
        filter_date = (
                wvc.query.Filter.by_property("valid_from").less_or_equal(query_date.strftime('%Y-%m-%dT%H:%M:%SZ')) &
                wvc.query.Filter.by_property("valid_to").greater_or_equal(query_date.strftime('%Y-%m-%dT%H:%M:%SZ'))
        )
        combined_filter = filter_general & filter_date

        # Execute query
        query_result = collection.query.fetch_objects(
            return_properties =  ["article_number", "article_content", "valid_from", "valid_to"],
            filters=combined_filter,
        )
        objects = query_result.objects
        batch_data = [
            {
                **{k: str(v) if hasattr(v, '__str__') and not isinstance(v,
                                                                         (str, int, float, bool, list, dict)) else v
                   for k, v in obj.properties.items()}
            }
            for obj in objects
        ]
        # st.write(batch_data)
        return batch_data

    except Exception as e:
        print(f"Error retrieving general terms: {str(e)}")
        return []


def perform_hybrid_search(client, query: str, query_date: datetime, limit: int = 5) -> List[Dict]:
    """Perform hybrid search with time-based filtering"""
    try:
        collection = client.collections.get("Vmi_docs")

        # Create date filter
        filter_date = (
                wvc.query.Filter.by_property("valid_from").less_or_equal(query_date.strftime('%Y-%m-%dT%H:%M:%SZ')) &
                wvc.query.Filter.by_property("valid_to").greater_or_equal(query_date.strftime('%Y-%m-%dT%H:%M:%SZ'))
        )

        # Execute hybrid search
        response = collection.query.hybrid(
            query=query,
            filters=filter_date,
            alpha=0.5,
            return_metadata=wvc.query.MetadataQuery(
                score=True,
                explain_score=True
            ),
            limit=limit
        )

        results = []
        for obj in response.objects:
            result = {
                "properties": obj.properties,
                "score": obj.metadata.score,
                "explain_score": obj.metadata.explain_score
            }
            results.append(result)

        return results

    except Exception as e:
        print(f"Error in hybrid search: {str(e)}")
        return []


def evaluate_legal_documents(query: str, general_terms: List[Dict], search_results: List[Dict], llm: ChatOpenAI) -> str:
    """Evaluate and format legal document results"""

    # Format documents for evaluation
    docs_list = "\n\nGeneral Terms:\n" + "\n\n".join([
        f"Article {doc.get('article_number', '')}\n"
        f"Valid from: {doc.get('valid_from', '')}\n"
        f"{doc.get('article_content', '')}"
        for doc in general_terms
    ])

    docs_list += "\n\nRelevant Articles:\n" + "\n\n".join([
        f"Article {result['properties'].get('article_number', '')}\n"
        f"Valid from: {result['properties'].get('valid_from', '')}\n"
        f"{result['properties'].get('article_content', '')}\n"
        f"Relevance score: {result.get('score', 0)}"
        for result in search_results
    ])

    evaluation_prompt = f"""
    Evaluate how the retrieved legal documents answer the user's query.
    Consider both general terms and specific articles.

    User query: {query}

    Available documents:
    {docs_list}

    Provide a structured response that:
    1. Cites relevant articles directly
    2. Explains how they answer the query
    3. Notes any missing information
    4. Mentions time period relevance

    Keep answers factual and based solely on the provided documents. Answer in Lithuanian language.
    """

    try:
        structured_llm = llm.with_structured_output(LegalDocumentEvaluation)
        evaluation = structured_llm.invoke(evaluation_prompt)

        # Format response
        response = "📋 Atsakymas į jūsų užklausą:\n\n"

        response += evaluation.relevance_explanation + "\n\n"

        response += "📜 Aktualūs straipsniai:\n"
        for citation in evaluation.cited_articles:
            response += f"• {citation}\n"

        if evaluation.missing_information:
            response += "\n⚠️ Papildoma informacija, kuri padėtų tiksliau atsakyti:\n"
            response += "\n".join(f"• {info}" for info in evaluation.missing_information)

        if evaluation.time_period_note:
            response += f"\n\n⏰ {evaluation.time_period_note}"

        return response

    except Exception as e:
        print(f"Error in document evaluation: {str(e)}")
        return "Atsiprašome, įvyko klaida vertinant dokumentus. Bandykite dar kartą."


def process_legal_query(prompt: str, client, llm: ChatOpenAI) -> Tuple[str, List[Dict]]:
    """Process legal query and return relevant documents"""

    # Step 1: Extract relevant date from query
    st.write(prompt)
    query_date = extract_date_from_query(prompt, llm)
    st.write(f"Query date: {query_date}")

    # Step 2: Get general terms
    general_terms = get_general_terms(client, query_date)
    st.write(f"Found {len(general_terms)} general terms articles")

    # Step 3: Perform hybrid search
    search_results = perform_hybrid_search(client, prompt, query_date)
    print(f"Found {len(search_results)} relevant articles")

    # Step 4: Evaluate and format results
    response = evaluate_legal_documents(prompt, general_terms, search_results, llm)

    return response, search_results


def main():
    st.title("🇱🇹 Teisinių dokumentų asistentas")

    init_session_state()

    try:
        client = setup_weaviate_client()
        llm = ChatOpenAI(model="gpt-4o", temperature=0.1)

    except Exception as e:
        st.error("Klaida jungiantis prie dokumentų duomenų bazės")
        return

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Užduokite klausimą apie teisinį reguliavimą..."):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Process query and display results
        with st.chat_message("assistant"):
            try:
                with st.spinner("Ieškau tinkamų dokumentų..."):
                    response, search_results = process_legal_query(prompt, client, llm)

                    # Add response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.markdown(response)

                    # Display debug information in sidebar if needed
                    with st.sidebar:
                        if st.checkbox("Rodyti detalią informaciją"):
                            st.json([{
                                "article_number": r['properties'].get('article_number'),
                                "valid_from": r['properties'].get('valid_from'),
                                "score": r['score']
                            } for r in search_results])

            except Exception as e:
                st.error("Įvyko klaida apdorojant užklausą")
                print(f"Error: {str(e)}")

    # Clear chat button
    with st.sidebar:
        if st.button("Išvalyti pokalbį"):
            st.session_state.messages = []
            st.session_state.conversation_memory.clear()
            st.rerun()


if __name__ == "__main__":
    main()
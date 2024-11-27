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
import pandas as pd

wcd_url = st.secrets["weaviate_credentials"]["url"]
weaviate_token = st.secrets["weaviate_credentials"]["token"]
openai_api_key = st.secrets["openai_credentials"]["api_key"]

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
    if "query_date" not in st.session_state:
        st.session_state.query_date = datetime.now()


def setup_weaviate_client():
    return weaviate.connect_to_weaviate_cloud(
        cluster_url=wcd_url,
        auth_credentials=Auth.api_key(weaviate_token),
        headers={"X-OpenAI-Api-Key": openai_api_key}
    )


def extract_date_from_query(query: str, llm: ChatOpenAI) -> datetime:
    """Extract date from query using LLM if no date is set in session"""
    if "query_date" not in st.session_state:
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
            st.session_state.query_date = datetime.strptime(result.content.strip(), '%Y-%m-%d')
        except:
            st.session_state.query_date = datetime.now()

    return st.session_state.query_date


def get_full_article(client, article_number: str, valid_from: str, valid_to: str) -> List:
    """Retrieve full article content by article number and validity period"""
    try:
        collection = client.collections.get("Vmi_docs")

        # Create filter for exact article match
        filter_article = wvc.query.Filter.by_property("article_number").equal(article_number)
        filter_date = (
                wvc.query.Filter.by_property("valid_from").equal(valid_from) &
                wvc.query.Filter.by_property("valid_to").equal(valid_to)
        )
        combined_filter = filter_article & filter_date

        # Execute query
        query_result = collection.query.fetch_objects(
            return_properties=[
                "article_number",
                "article_content",
                "article_title",
                "chapter_title",
                "valid_from",
                "valid_to",
                "name",
                "doc_link"
            ],
            filters=combined_filter,
        )

        if query_result.objects:
            return [obj.properties for obj in query_result.objects]
        return []

    except Exception as e:
        print(f"Error retrieving full article: {str(e)}")
        return []


def get_general_terms(client, query_date: datetime) -> List[Dict]:
    """Retrieve general terms articles valid for the given date"""
    try:
        collection = client.collections.get("Vmi_docs")

        filter_general = wvc.query.Filter.by_property("article_number").contains_any(["1", "2"])
        filter_date = (
                wvc.query.Filter.by_property("valid_from").less_or_equal(query_date.strftime('%Y-%m-%dT%H:%M:%SZ')) &
                wvc.query.Filter.by_property("valid_to").greater_or_equal(query_date.strftime('%Y-%m-%dT%H:%M:%SZ'))
        )
        combined_filter = filter_general & filter_date

        query_result = collection.query.fetch_objects(
            return_properties=["article_number", "article_content", "article_title", "valid_from", "valid_to", "name",
                               "doc_link"],
            filters=combined_filter,
        )

        return [obj.properties for obj in query_result.objects]

    except Exception as e:
        print(f"Error retrieving general terms: {str(e)}")
        return []


def perform_hybrid_search(client, query: str, query_date: datetime, limit: int = 5) -> List[Dict]:
    """Perform hybrid search with time-based filtering"""
    try:
        collection = client.collections.get("Vmi_docs")

        filter_date = (
                wvc.query.Filter.by_property("valid_from").less_or_equal(query_date.strftime('%Y-%m-%dT%H:%M:%SZ')) &
                wvc.query.Filter.by_property("valid_to").greater_or_equal(query_date.strftime('%Y-%m-%dT%H:%M:%SZ'))
        )

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

        # Get full articles for each result
        full_results = []
        # print(response.objects)
        for obj in response.objects:
            properties = obj.properties
            print({properties.get('article_number'),
                   })
            full_article = get_full_article(
                client,
                properties.get('article_number'),
                properties.get('valid_from'),
                properties.get('valid_to')
            )
            if full_article:
                full_results += full_article

        return full_results

    except Exception as e:
        print(f"Error in hybrid search: {str(e)}")
        return []


def display_document_card(doc: Dict):
    """Display a document card with article information"""
    with st.container():
        st.markdown(f"""
        <div style='border: 1px solid #ddd; padding: 1rem; margin: 1rem 0; border-radius: 5px;'>
            <h4>{doc.get('article_title', '')}</h4>
            <p><strong>Straipsnis {doc.get('article_number', '')}: {doc.get('article_title', '')}</strong></p>
            <p><em>{doc.get('chapter_title', '')}</em></p>
            <p><small>Galioja nuo: {doc.get('valid_from', '').strftime('%Y-%m-%d')} iki {doc.get('valid_to', '').strftime('%Y-%m-%d')}</small></p>
            <a href="{doc.get('doc_link', '#')}" target="_blank">Originalus dokumentas →</a>
        </div>
        """, unsafe_allow_html=True)



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
        f"Article {result.get('article_number', '')}\n"
        f"Valid from: {result.get('valid_from', '')}\n"
        f"{result.get('article_content', '')}\n"
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

    Keep answers factual and based solely on the provided documents. Answer must be provided in Lithuanian language.
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


def process_legal_query(prompt: str, client, llm: ChatOpenAI):
    """Process legal query and return relevant documents"""
    query_date = st.session_state.query_date
    general_terms = get_general_terms(client, query_date)

    search_results = perform_hybrid_search(client, prompt, query_date)

    response = evaluate_legal_documents(prompt, general_terms, search_results, llm)
    return response, search_results, general_terms


def main():
    st.title("🇱🇹 Teisinių dokumentų asistentas")

    init_session_state()

    try:
        client = setup_weaviate_client()
        llm = ChatOpenAI(model="gpt-4o", temperature=0.1, api_key=openai_api_key)
    except Exception as e:
        st.error("Klaida jungiantis prie dokumentų duomenų bazės")
        return

    # Date selector in sidebar
    with st.sidebar:
        st.write("### Datos nustatymai")
        selected_date = st.date_input(
            "Pasirinkite datą dokumentų paieškai",
            value=st.session_state.query_date
        )
        st.session_state.query_date = datetime.combine(selected_date, datetime.min.time())

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Užduokite klausimą apie teisinį reguliavimą..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Ieškau tinkamų dokumentų..."):
                    response, search_results, general_terms = process_legal_query(prompt, client, llm)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.markdown(response)

                    docs = []
                    for doc in general_terms:
                        docs.append({"article_number": doc.get('article_number', ''),
                                     "article_title": doc.get('article_title', ''),
                                     "doc_link": doc.get('doc_link', ''),
                                     "valid_from": doc.get('valid_from', ''),
                                     "valid_to": doc.get('valid_to', '')})

                    for doc in search_results:
                        docs.append({"article_number": doc.get('article_number', ''),
                                     "article_title": doc.get('article_title', ''),
                                     "doc_link": doc.get('doc_link', ''),
                                     "valid_from": doc.get('valid_from', ''),
                                     "valid_to": doc.get('valid_to', '')})

                    df = pd.DataFrame(docs).drop_duplicates()
                    unique_articles = df.to_dict('records')

                    # Display document cards
                    st.write("### 📚 Bendrosios nuostatos ir Aktualūs straipsniai")
                    # st.write(general_terms)
                    for doc in unique_articles:
                        display_document_card(doc)


            except Exception as e:
                st.error("Įvyko klaida apdorojant užklausą")
                st.error(f"Error: {str(e)}")

    with st.sidebar:
        if st.button("Išvalyti pokalbį"):
            st.session_state.messages = []
            st.session_state.conversation_memory.clear()
            st.rerun()


if __name__ == "__main__":
    main()
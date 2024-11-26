#%%
import pandas as pd
import os
import weaviate
from openai import OpenAI
from weaviate.classes.config import Configure
import weaviate.classes as wvc
from langchain_community.document_loaders import BSHTMLLoader
from typing import Dict, List, Optional
from dataclasses import dataclass
from uuid import uuid4
from dataclasses import dataclass
import re
from typing import List


html_files_dir = 'vmi_docs'
collection_name = 'Vmi_docs'
csv_file = 'vmi_docs.csv'
wcd_url = 'https://xh1j9trzu5cervreztxw.c0.europe-west3.gcp.weaviate.cloud'
wcd_api_key = 'uTTyayyrfwyn98zBq6ukAcIAVnEJjkBWMLac'
openai_api_key = os.environ["OPENAI_API_KEY"]

weaviate_client = weaviate.connect_to_weaviate_cloud(
    cluster_url=wcd_url,
    auth_credentials=wvc.init.Auth.api_key(wcd_api_key),
    headers={"X-OpenAI-Api-Key": openai_api_key}
)

client = OpenAI()

#%%
@dataclass
class Chapter:
    id: str
    number: str          # Roman numeral
    type: str           # "SKYRIUS"
    title: str          # Chapter title
    content: str        # Full content
    raw_header: str     # Original header text

def split_into_chapters(text: str) -> List[Chapter]:
    """
    Split text into chapters based on 'SKYRIUS' headers.
    Returns list of Chapter objects.

    Example input:
    I SKYRIUS
    BENDROSIOS NUOSTATOS
    [chapter content...]
    II SKYRIUS
    KITOS NUOSTATOS
    [chapter content...]
    """
    # Pattern to match Roman numerals followed by SKYRIUS and title
    chapter_pattern = r'([IVX]+)\s*SKYRIUS\s*\n\s*(?:\*\*)?([^*\n]+)(?:\*\*)?\s*\n'

    # Find all chapter starts
    chapter_starts = list(re.finditer(chapter_pattern, text, re.MULTILINE))
    chapters = []

    # Process each chapter
    for i, match in enumerate(chapter_starts):
        chapter_id = str(uuid4())
        chapter_number = match.group(1)
        chapter_title = match.group(2).strip()
        chapter_header = match.group(0)

        # Calculate chapter content (from this chapter start to next chapter start or end of text)
        content_start = match.end()
        if i < len(chapter_starts) - 1:
            content_end = chapter_starts[i + 1].start()
        else:
            content_end = len(text)

        chapter_content = text[content_start:content_end].strip()

        chapter = Chapter(
            id=chapter_id,
            number=chapter_number,
            type="SKYRIUS",
            title=chapter_title,
            content=chapter_content,
            raw_header=chapter_header.strip()
        )
        chapters.append(chapter)

    return chapters
#%%
def clean_text(text: str) -> str:
    """
    Clean text by removing article adjustments, end document text, and normalize whitespace.
    """
    # Define patterns to remove
    amendment_pattern = r'(?:Papildyta straipsnio punktu:|Straipsnio dalies pakeitimai:|' \
                       r'Papildyta straipsniu:|Papildyta straipsnio dalimi:|' \
                       r'Straipsnio dalies numeracijos pakeitimas:|Straipsnio pakeitimai:)' \
                       r'\s+(?:Nr\.[^\n]+(?:\([^\)]+\))?\s*)+'
    cleaned_text = re.sub(amendment_pattern, '', text, flags=re.MULTILINE)

    # Remove all adjustment patterns

    # Remove everything after the end marker
    end_marker = "Skelbiu šį Lietuvos Respublikos Seimo priimtą įstatymą."
    if end_marker in cleaned_text:
        cleaned_text = cleaned_text.split(end_marker)[0]

    # # Remove double whitespaces and normalize newlines
    cleaned_text = re.sub(r'[ ]+', ' ', cleaned_text)  # Replace multiple spaces with single space
    cleaned_text = re.sub(r'\n+', ' \n', cleaned_text)  # Normalize multiple newlines
    cleaned_text = cleaned_text.strip()

    return cleaned_text
#%%
@dataclass
class Article:
    id: str
    number: str          # Article number
    title: str          # Article title
    content: str        # Full content
    raw_header: str     # Original header text

def split_into_articles(text: str) -> List[Article]:
    """
    Split text into articles based on "straipsnis" headers.
    Example: "1 straipsnis. Įstatymo paskirtis ir taikymo sritis"
    """
    # Pattern to find article headers with lookahead for next article or end of string
    article_pattern = r'(\d+)\s+straipsnis\.\s*([^\n]+)'

    # Find all matches of article headers
    matches = list(re.finditer(article_pattern, text))
    articles = []

    for i in range(len(matches)):
        current_match = matches[i]
        article_id = str(uuid4())
        article_number = current_match.group(1)
        article_title = current_match.group(2).strip()
        article_header = current_match.group(0)

        # Get content until next article or end of text
        start_pos = current_match.end()
        if i < len(matches) - 1:
            end_pos = matches[i + 1].start()
        else:
            end_pos = len(text)

        # Extract content
        content = text[start_pos:end_pos].strip()

        # Create Article object
        article = Article(
            id=article_id,
            number=article_number,
            title=article_title,
            content=content,
            raw_header=article_header
        )
        articles.append(article)

    return articles


#%%
def parse_document(content:str) -> List:
    chapters_list = split_into_chapters(content)
    # print(chapters_list[1])
    articles = []
    for chapter in chapters_list:
        cleaned_chapter_text = clean_text(chapter.content)
        chapter_articles = split_into_articles(cleaned_chapter_text)

        for article in chapter_articles:
            print(f"  Article {article.number}: {article.title}")
            articles.append({
                "chapter_id": chapter.id,
                "chapter_number": chapter.number,
                "chapter_type": chapter.type,
                "chapter_title": chapter.title,
                "chapter_raw_header": chapter.raw_header,
                "article_id": article.id,
                "article_number": article.number,
                "article_title": article.title,
                "article_content": article.content,
                "article_raw_header": article.raw_header
            })

    return articles
#%%
def create_weaviate_schema(collection_name):
    """Create collection in Weaviate with enhanced hierarchical structure"""
    from weaviate.classes.config import Property, DataType
    try:
        if weaviate_client.collections.exists(collection_name):
            print("Collection already exists")
            return weaviate_client.collections.get(collection_name)
        properties = [
            Property(name="doc_link", data_type=DataType.TEXT),
            Property(name="name", data_type=DataType.TEXT),
            Property(name="valid_to", data_type=DataType.DATE),
            Property(name="version", data_type=DataType.TEXT),
            Property(name="doc_id", data_type=DataType.TEXT),
            Property(name="article_number", data_type=DataType.TEXT),
            Property(name="valid_from", data_type=DataType.DATE),
            Property(name="chapter_number", data_type=DataType.TEXT),
            Property(name="article_title", data_type=DataType.TEXT),
            Property(name="chapter_id", data_type=DataType.UUID),
            Property(name="chapter_raw_header", data_type=DataType.TEXT),
            Property(name="url", data_type=DataType.TEXT),
            Property(name="article_raw_header", data_type=DataType.TEXT),
            Property(name="chapter_title", data_type=DataType.TEXT),
            Property(name="article_content", data_type=DataType.TEXT),
            Property(name="chapter_type", data_type=DataType.TEXT),
            Property(name="article_id", data_type=DataType.TEXT),
        ]

        collection = weaviate_client.collections.create(
            name=collection_name,
            properties=properties,
            vectorizer_config=Configure.Vectorizer.text2vec_openai(),
            generative_config=Configure.Generative.openai()
        )
        print("Collection created successfully")
        return collection

    except Exception as e:
        print(f"Error creating schema: {str(e)}")
        raise

def upload_to_weaviate(input_dir=html_files_dir):
    """Upload restructured data to Weaviate"""
    try:
        collection = create_weaviate_schema(collection_name)
        df = pd.read_csv(csv_file)
        print(f"Processing {len(df)} records")
        i = 0

        with collection.batch.fixed_size(batch_size=200) as batch:
            for idx, row in df.iterrows():
                try:
                    metadata = {
                        "name": str(row['name']),
                        "valid_from": str(row['valid_from']),
                        "valid_to": str(row['valid_to']),
                        "doc_link": str(row['doc_link']),
                        "doc_id": str(row['id']),
                        "version": str(row['version']),
                        "url": str(row['url'])
                    }

                    file_path = os.path.join(input_dir, f"{metadata['version']}.html")
                    if not os.path.exists(file_path):
                        print(f"File not found: {file_path}")
                        continue

                    # Load and process HTML content
                    html_loader = BSHTMLLoader(file_path, bs_kwargs={'features': 'html.parser'})
                    documents = html_loader.load()
                    content = "\n".join([doc.page_content for doc in documents])

                    # Parse document into hierarchical structure
                    articles = parse_document(content)
                    print("successfully parsed doc")
                    # Upload each structured section
                    for section in articles:
                        document_object = {
                            **metadata,
                            **section
                        }
                        print(section)
                        print( {
                            "chapter_raw_header": section['chapter_raw_header'],
                            "article_number": section['article_number'],
                            "article_raw_header": section['article_raw_header'],
                            "version": metadata['version']})

                        batch.add_object(properties=document_object )
                        i = i + 1
                        print(i)

                except Exception as e:
                    print(f"Error processing record {idx}: {str(e)}")
                    continue

    except Exception as e:
        print(f"Error in upload process: {str(e)}")
        raise



#%%
collection = create_weaviate_schema(collection_name)
#%%
df = pd.read_csv(csv_file)
#%%
df = df[0:20]
#%%
input_dir='vmi_docs'
#%%
i = 0

from datetime import datetime

def format_date_rfc3339(date_str):
   date_obj = datetime.strptime(date_str, '%Y-%m-%d')
   return date_obj.strftime('%Y-%m-%dT%H:%M:%SZ')
from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_text(text, chunk_size=4000):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text)

i = 0
with collection.batch.fixed_size(batch_size=100) as batch:
    for idx, row in df.iterrows():
        metadata = {
            "name": str(row['name']),
            "valid_from": str(row['valid_from']),
            "valid_to": str(row['valid_to']),
            "doc_link": str(row['doc_link']),
            "doc_id": str(row['id']),
            "version": str(row['version']),
            "url": str(row['url'])
        }

        file_path = os.path.join(input_dir, f"{metadata['version']}.html")
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue

        html_loader = BSHTMLLoader(file_path, bs_kwargs={'features': 'html.parser'})
        documents = html_loader.load()
        content = "\n".join([doc.page_content for doc in documents])
        articles = parse_document(content)
        print("successfully parsed doc")

        for section in articles:
            # Split article content into chunks
            content_chunks = chunk_text(section["article_content"])

            for chunk in content_chunks:
                document_object = {
                    **metadata,
                    **section,
                    "article_content": chunk  # Replace original content with chunk
                }
                document_object['valid_from'] = format_date_rfc3339(document_object['valid_from'])
                document_object['valid_to'] = format_date_rfc3339(document_object['valid_to'])
                print(document_object)

                batch.add_object(properties=document_object)
                i += 1
                print(i)
#%%
from datetime import datetime

def format_date_rfc3339(date_str):
   date_obj = datetime.strptime(date_str, '%Y-%m-%d')
   return date_obj.strftime('%Y-%m-%dT%H:%M:%SZ')

# Upload each structured section
section = articles[0]
document_object = {
    **metadata,
    **section
}
document_object['valid_from'] = format_date_rfc3339(document_object['valid_from'])
document_object['valid_to'] = format_date_rfc3339(document_object['valid_to'])
    # print(document_object)
    # print( {
    #     "chapter_raw_header": section['chapter_raw_header'],
    #     "article_number": section['article_number'],
    #     "article_raw_header": section['article_raw_header'],
    #     "version": metadata['version']})

    # batch.add_object(properties=document_object )
#%%

#%%
with collection.batch.fixed_size(batch_size=100) as batch:
    print(document_object)
    batch.add_object(properties=document_object )
#%%
collection.batch.failed_objects
#%%
upload_to_weaviate(input_dir="vmi_docs")
#%%

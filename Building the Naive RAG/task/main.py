import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_openai import OpenAI, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

load_dotenv()
from langchain_community.document_loaders import IMSDbLoader
from bs4 import BeautifulSoup
import re
import requests

response = requests.get("https://imsdb.com/all-scripts.html")
soup = BeautifulSoup(response.text, "html.parser")

links = soup.find_all("a")
movie_titles = []
for link in links:
    href = link.get("href", "")
    if href.__contains__("/Movie Scripts/"):
        if not link.text.__contains__("\n"):
            movie_titles.append(link.text)

for i, movie in enumerate(movie_titles, start=1):
    print(f"{i}. {movie}")

query = input()

if movie_titles.__contains__(query):
    print(str(movie_titles.index(query)) + '. ' + query)
    url = f"https://imsdb.com/scripts/{query.replace(' ', '-')}.html."
    loader = IMSDbLoader(url)
    script = loader.load()

    cleaned_text = re.sub(r'\s+', ' ', script[0].page_content).strip()

    splitter = RecursiveCharacterTextSplitter(
        separators=["INT."],
        chunk_size=500,
        chunk_overlap=10
    )

    scenes = splitter.create_documents([cleaned_text])

    print(f"Loaded script for {query} from {url}.")
    print(f"Found {len(scenes)} scenes in the script for {query}.")

    embeddings = HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
    )

    collection_name = query.replace(' ', '-')
    client = QdrantClient(url="http://localhost:6333")
    client.collection_exists(collection_name)
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=384,
            distance=Distance.COSINE
        )
    )

    QdrantVectorStore.from_documents(
        documents=scenes,
        embedding=embeddings,
        url="http://localhost:6333",
        collection_name=collection_name,
    )

    print(f"Embedded script for {query}.")

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings
    )

    user_input = input()

    context = "\n\n".join([scene.page_content for scene in scenes])

    prompt_template = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an expert on movie scripts with deep knowledge of 
            cinematic storytelling, character development, and scene 
            construction. Use the following retrieved scenes from the 
            movie script as context to answer the user's query in detail.

            Context:
            {context}
            """
        ),
        (
            "human",
            "{question}"
        )
    ])

    llm = ChatOpenAI(
        api_key=os.getenv("OPEN_API_KEY"),
        model="gpt-4o-mini",
        base_url="https://litellm.aks-hs-prod.int.hyperskill.org/openai",
    )

    chain = prompt_template | llm

    response = chain.invoke({
        "context": context,
        "question": user_input
    })

    print(f"\nFinal Answer:\n{response.content}")
else:
    print(f"Script for '{query}' wasn't found in the list of movie scripts.")

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter

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
    print(cleaned_text)

    splitter = RecursiveCharacterTextSplitter(
        separators=["INT."],
        chunk_size=500,
        chunk_overlap=10
    )

    scenes = splitter.create_documents([cleaned_text])

    print(f"Loaded script for {query} from {url}.")
    print(f"Found {len(scenes)} scenes in the script for {query}.")

    for i, scene in enumerate(scenes, start=1):
        print(f"Scene {i}: {scene.page_content}")
else:
    print(f"Script for '{query}' wasn't found in the list of movie scripts.")

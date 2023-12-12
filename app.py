"""##Initialize##"""

import streamlit as st
import streamlit.components.v1 as components
import requests

import os
import json
from enum import Enum
from bs4 import BeautifulSoup, NavigableString
from llama_index.indices.multi_modal.base import MultiModalVectorStoreIndex
from llama_index.vector_stores import Neo4jVectorStore
from llama_index import StorageContext, Document
from llama_index.schema import ImageDocument
from llama_index.node_parser import SimpleNodeParser
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from llama_index.graph_stores import Neo4jGraphStore
from llama_index.retrievers import KnowledgeGraphRAGRetriever
from llama_index.query_engine import RetrieverQueryEngine
import tiktoken
import seaborn as sns
import requests
from PIL import Image
import matplotlib.pyplot as plt
from io import BytesIO
from IPython.display import Markdown, display
from pydantic import BaseModel
from IPython.display import HTML, display
from llama_index.prompts import PromptTemplate
from llama_index.query_engine import SimpleMultiModalQueryEngine
from llama_index.indices.multi_modal.retriever import MultiModalVectorIndexRetriever

os.environ["OPENAI_API_KEY"] = "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"
NEO4J_URI="neo4j+s://e27fadb9.databases.neo4j.io"
NEO4J_USERNAME="neo4j"
NEO4J_PASSWORD="4vBuqs-oiY_VjxNJumaWt0NByCKmVDoIn27zJ7fmw_8"

from llama_index import (
    SimpleDirectoryReader,
    LLMPredictor,
    ServiceContext,
    KnowledgeGraphIndex,
)

from llama_index.llms import OpenAI

llm = OpenAI(temperature=0, model="gpt-4-1106-preview")

service_context = ServiceContext.from_defaults(llm=llm, chunk_size=512)

def display_first_twitter_link(video_nodes):
    if video_nodes:
        video_url = video_nodes[0].node.metadata["video_url"]
        response = requests.get(f"https://publish.twitter.com/oembed?url={video_url}&omit_script=true")
        if response.status_code == 200:
            tweet_html = response.json()["html"]
            components.html(tweet_html, height=700)

def plot_first_image(image_nodes):
    if image_nodes:
        img_url = image_nodes[0].node.image_url
        try:
            response = requests.get(img_url)
            response.raise_for_status()  # Raise an error for bad status codes
            image = Image.open(BytesIO(response.content))

            plt.figure(figsize=(20, 10))  # Adjust the figure size as needed
            plt.imshow(image)
            plt.xticks([])
            plt.yticks([])
            plt.show()
        except Exception as e:
            print(f"Error loading image {img_url}: {e}")

def process_html_file(file_path):
    print(f"Processing file: {file_path}")
    with open(file_path, "r", encoding="utf-8") as file:
        soup = BeautifulSoup(file, "html.parser")

    main_content = soup

    sections = []
    current_section = {"header": "", "content": "", "source": file_path.split("/")[-1]}
    images = []
    videos = []
    header_found = False

    for element in main_content.find_all(recursive=True):
        if element.name in ["h1", "h2", "h3", "h4"]:
            if header_found and current_section["content"].strip():
                sections.append(current_section)
            current_section = {
                "header": element.get_text(),
                "content": "",
                "source": file_path.split("/")[-1],
            }
            header_found = True
        elif header_found:
            if element.name == "p":
                current_section["content"] += element.get_text().strip() + "\n"
            elif element.name == "img":
                img_src = element.get("src")
                images.append({"url": img_src, "alt": element.get("alt"), "source": file_path.split("/")[-1]})
            elif element.name == "figure" and "wp-block-embed-twitter" in element.get("class", []):
                tweet_link = element.find('a')
                if tweet_link:
                    tweet_url = tweet_link.get("href")
                    videos.append({"url": tweet_url, "type": "twitter", "source": file_path.split("/")[-1]})

    if current_section["content"].strip():
        sections.append(current_section)

    # Create a list of tuples (image, associated_video_url)
    image_video_pairs = []
    for image in images:
        # Find the video URL from the same section if available
        associated_video_url = next((video['url'] for video in videos if video['source'] == image['source']), None)
        image_video_pairs.append((image, associated_video_url))

    print(f"Returning {len(images)} images, {len(sections)} sections, and {len(videos)} videos")
    return image_video_pairs, sections, videos

# Directory to search in (current working directory)
directory = os.getcwd()

all_documents = []
all_images = []
all_videos = []

# Walking through the directory
for root, dirs, files in os.walk(directory):
    for file in files:
        if file.endswith(".html"):
            image_video_pairs, sections, videos = process_html_file(os.path.join(root, file))
            all_documents.extend(sections)
            for image, video_url in image_video_pairs:
                image['video_url'] = video_url  # Add video URL to image metadata
                all_images.append(image)
            all_videos.extend(videos)

text_docs = [Document(text=el.pop("content"), metadata=el) for el in all_documents]
image_docs = [
    ImageDocument(
        image_url=img['url'],
        text=f"{img['source']} {img['alt']}",  # Concatenate the source file name with the alt text
        metadata={
            'alt': img['alt'],
            'source': img['source'],
            'video_url': img.get('video_url'),
            # Include any other metadata as needed
        }
    ) for img in all_images
]

# Assuming you want to print counts of each type
print(f"Text document count: {len(text_docs)}")
print(f"Image document count: {len(image_docs)}")
print(f"Video count: {len(all_videos)}")

# Print the documents if needed
# print(video_docs)
# print(text_docs)
print(image_docs)

text_store = Neo4jVectorStore(
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
    index_name="text_collection",
    node_label="Chunk",
    embedding_dimension=1536
)
image_store = Neo4jVectorStore(
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
    index_name="image_collection",
    node_label="Image",
    embedding_dimension=512
)

storage_context = StorageContext.from_defaults(vector_store=text_store)

index = MultiModalVectorStoreIndex.from_documents(
    text_docs + image_docs,  # Adjusted to use the new lists
    storage_context=storage_context,
    image_vector_store=image_store,
)

openai_mm_llm = OpenAIMultiModal(
    model="gpt-4-vision-preview", max_new_tokens=2000
)


qa_tmpl_str = (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer the query.\n"
    "Query: {query_str}\n"
    "Answer: "
)
qa_tmpl = PromptTemplate(qa_tmpl_str)

query_engine = index.as_query_engine(
    multi_modal_llm=openai_mm_llm,
    text_qa_template=qa_tmpl,
    verbose=True,
    include_text=True,
    response_mode="tree_summarize",
    embedding_mode="hybrid",
    similarity_top_k=15,
    image_similarity_top_k=1,
)

"""##Streamlit##"""

def main():
    st.title("Query Processing App")

    # Input for the query
    query_str = st.text_input("Enter your query:", "Tell me about Dylan Crews. What are his strengths and weaknesses?")

    if st.button("Run Query"):
        with st.spinner("Processing..."):
            # Run the program
            response = query_engine.query(query_str)

            # Return the text
            st.markdown(f"**Response:**\n{response}")

            # Display the image
            image_nodes = response.metadata.get("image_nodes")
            if image_nodes:
                img_url = image_nodes[0].node.image_url
                st.image(img_url, caption="Image related to the query")

            # Display the Twitter URL in an embedded Twitter widget
            video_nodes = response.metadata.get("video_nodes")
            if video_nodes:
                display_first_twitter_link(video_nodes)

if __name__ == "__main__":
    main()
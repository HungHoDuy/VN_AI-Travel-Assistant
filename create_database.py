# from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader
from transformers import AutoTokenizer, AutoModel
import torch
# from langchain.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os
import shutil
import glob
import json



CHROMA_PATH = "chroma"
DATA_PATH = "data/"





def load_documents(file_path: str):
    """
    Load and process travel data (hotels and restaurants) from a JSON file into LangChain Document objects.
    
    Args:
        file_path (str): Path to the JSON file containing travel data.
    
    Returns:
        list[Document]: A list of Document objects containing hotels and restaurants data.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        json_data = json.load(file)

    documents = []
    hotels = json_data.get("hotels", {}).get("hotels", [])

    # Process hotel data
    for hotel in hotels:
        # Extract name and description for embedding
        page_content = f"Name: {hotel['name']}\nDescription: {hotel['description']}"

        # Convert room types to a comma-separated string
        room_names = ", ".join([room['name'] for room in hotel['room_types']])
        
        # Store other details in metadata
        metadata = {
            "type": "hotel",
            "source": "travel_data_enhanced.json",
            "location": hotel['location'],
            "amenities": ", ".join(hotel['amenities']),  # Converting list to string
            "room_types": room_names  # Room types as a string of names
        }


        documents.append(Document(page_content=page_content, metadata=metadata))

    # Process restaurant data
    restaurants = json_data.get("restaurants", {}).get("restaurants", [])
    for restaurant in restaurants:
        # Extract name and description for embedding
        page_content = f"Name: {restaurant['name']}\nDescription: {restaurant['description']}"

        # Convert price range dictionary to a string
        price_range_str = f"{restaurant['price_range']['min']} - {restaurant['price_range']['max']} {restaurant['price_range']['currency']}"

        # Store other details in metadata
        metadata = {
            "type": "restaurant",
            "source": "travel_data_enhanced.json",
            "address": restaurant['address'],
            "phone": restaurant['phone'],
            "cuisine": ", ".join(restaurant['cuisine']),  # Convert list to string
            "features": ", ".join(restaurant['features']),  # Convert list to string
            "price_range": price_range_str,  # Price range as a string
            "opening_hours": ", ".join([f"{day}: {hours}" for day, hours in restaurant['opening_hours'].items()])  # Format opening hours
        }

        documents.append(Document(page_content=page_content, metadata=metadata))
    return documents


def save_to_chroma(documents: list[Document]):
    """
    Save documents to a Chroma vector store.

    Args:
        documents (list[Document]): A list of Document objects to store.
    """
    # Clear out the database first.
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Initialize Hugging Face embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    # Create a new DB from the documents.
    db = Chroma.from_documents(
        documents, embeddings, persist_directory=CHROMA_PATH
    )
    db.persist()
    print(f"Saved {len(documents)} documents to {CHROMA_PATH}.")


if __name__ == "__main__":
    file_path = "data\\travel_data_enhanced.json"
    documents = load_documents(file_path)
    print(f"Loaded {len(documents)} documents.")
    save_to_chroma(documents)
from sentence_transformers import SentenceTransformer
from langchain_chroma import Chroma
from langchain_community.llms.ollama import Ollama
CHROMA_PATH = "chroma"

class CustomEmbeddingFunction:
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)

    def embed_query(self, text):
        return self.model.encode(text, convert_to_tensor=True).tolist()  # Convert to list for compatibility

    def embed_documents(self, texts):
        return self.model.encode(texts, convert_to_tensor=True).tolist()  # Convert to list for compatibility

def query_rag(query_text: str):
    # Prepare the DB.
    embedding_function = CustomEmbeddingFunction('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=5)

    # Prepare context text with source information from results
    context_text = "\n\n---\n\n".join(
    [
        f"Type: {doc.metadata.get('type', 'Unknown')}\n"
        f"Location: {doc.metadata.get('location', 'Not available')}\n"
        f"Amenities: {doc.metadata.get('amenities', 'Not available')}\n"
        f"Room Types: {doc.metadata.get('room_types', 'Not available')}\n"
        f"Content: {doc.page_content}\n"
        f"Address: {doc.metadata.get('address', 'Not available')}\n"
        f"Phone: {doc.metadata.get('phone', 'Not available')}\n"
        f"Cuisine: {doc.metadata.get('cuisine', 'Not available')}\n"
        f"Features: {doc.metadata.get('features', 'Not available')}\n"
        f"Price Range: {doc.metadata.get('price_range', 'Not available')}\n"
        f"Opening Hours: {doc.metadata.get('opening_hours', 'Not available')}\n"
        f"Score: {_score:.4f}"  # Add the score with 4 decimal places
        for doc, _score in results
    ]
    )


    # Output the context text with source information
    return context_text

if __name__ == "__main__":
    
    query = "Giới thiệu cho tôi một vài khách sạn không hút thuốc ở Quy Nhơn"
    print(query_rag(query))


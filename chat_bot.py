from query_data import query_rag
from langchain_ollama import OllamaLLM
from langchain.prompts import ChatPromptTemplate

query = "Giới thiệu cho tôi một vài khách sạn không hút thuốc ở Quy Nhơn"
context_text = query_rag(query)

PROMPT_TEMPLATE = """
Use Chain of thought in English
Answer the question based only on the context that queried from RAG system:

{context}

---

Answer the question based on the above context: {question}
The final answer should be FULLY translated into the question language
"""

# Combine the context and the query to form the prompt    
prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE) 
prompt = prompt_template.format(context=context_text, question=query)

model = OllamaLLM(model="deepseek-r1:7b", streaming=True)  # Enable streaming

def generate_response():
    for chunk in model.stream(prompt):  # Streaming response
        yield chunk  # Yield each word/token as it arrives

# Example: Printing streamed response
for word in generate_response():
    print(word, end="", flush=True)  # Print without newline for a streaming effect

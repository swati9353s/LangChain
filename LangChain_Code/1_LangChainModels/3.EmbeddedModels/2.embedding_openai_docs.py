from langchain_openai import OpenAIEmbeddings   # Embedding interface (OpenAI-compatible)
from dotenv import load_dotenv                 # To load environment variables from .env file
import os

# Load API keys and base URL from .env file
load_dotenv()

# Create the embedding model connection
# We are using a Hugging Face embedding model through our custom LLM server
embedding = OpenAIEmbeddings(
    model='intfloat/multilingual-e5-large',    # Embedding model name
    api_key=os.getenv('LLM_API_KEY'),          # API key for the LLM server
    base_url=os.getenv('LLM_BASE_URL')         # Base URL of the LLM server
)

# List of text documents that we want to convert into embeddings
documents = [
    "Delhi is the capital of India",
    "Kolkata is the capital of West Bengal"
]

# Convert all documents into embedding vectors
# Each sentence will become a list of numerical values
result = embedding.embed_documents(documents)

# Print the embeddings (list of vectors)c
print(str(result))

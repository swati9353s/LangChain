from langchain_openai import OpenAIEmbeddings   # Embedding interface (OpenAI-style, but can work with any compatible server)
from dotenv import load_dotenv                 # To load API keys and URLs from .env file
import os

# Load environment variables from .env file
load_dotenv()

# Create an embedding model connection
# Even though the class name says "OpenAIEmbeddings",
# we are actually connecting to our own LLM server (not OpenAI)
embedding = OpenAIEmbeddings(
    model='intfloat/multilingual-e5-large',   # Hugging Face embedding model
    api_key=os.getenv('LLM_API_KEY'),         # API key for your custom LLM server
    base_url=os.getenv('LLM_BASE_URL')        # Base URL of your LLM server
)

# Convert text into a numerical vector (embedding)
result = embedding.embed_query("Hello world")

# Print the embedding vector (list of numbers)
print(str(result))

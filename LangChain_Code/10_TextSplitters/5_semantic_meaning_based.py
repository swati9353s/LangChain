# Import SemanticChunker
# SemanticChunker splits text based on MEANING (semantic similarity)
# instead of fixed character length.
from langchain_experimental.text_splitter import SemanticChunker

# OpenAIEmbeddings converts text into vector embeddings.
# These embeddings help measure semantic similarity between sentences.
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings 
import os

# Used to load environment variables (.env file)
# Typically contains OPENAI_API_KEY
from dotenv import load_dotenv

# Load API keys and environment variables
load_dotenv()

# open ai embedding model is not working
embedding = OpenAIEmbeddings(
    model='intfloat/multilingual-e5-large',    # Embedding model name
    api_key=os.getenv('LLM_API_KEY'),          # API key for the LLM server
    base_url=os.getenv('LLM_BASE_URL')         # Base URL of the LLM server
)

#Hugging face embedding model
embedding1 = HuggingFaceEmbeddings(
    model_name='sentence-transformers/all-MiniLM-L6-v2'
)

#Open ai embedding model of hugging face

# ---------------------------------------------------------
# Create Semantic Chunker
# ---------------------------------------------------------
# OpenAIEmbeddings() → converts sentences into vectors
#
# breakpoint_threshold_type="standard_deviation"
#   → decides where to split text based on how different
#     neighbouring sentences are semantically.
#
# breakpoint_threshold_amount=3
#   → higher value = fewer splits (only large topic changes create chunks)
#   → lower value = more splits
#
# The splitter will automatically detect topic changes.
text_splitter = SemanticChunker(
    embedding1,
    breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=3
)


# ---------------------------------------------------------
# Sample text containing multiple topics
# (farming → cricket → terrorism)
# SemanticChunker should separate these into meaningful chunks
# ---------------------------------------------------------
sample = """
Farmers were working hard in the fields, preparing the soil and planting seeds for the next season. The sun was bright, and the air smelled of earth and fresh grass. The Indian Premier League (IPL) is the biggest cricket league in the world. People all over the world watch the matches and cheer for their favourite teams.

Terrorism is a big danger to peace and safety. It causes harm to people and creates fear in cities and villages. When such attacks happen, they leave behind pain and sadness. To fight terrorism, we need strong laws, alert security forces, and support from people who care about peace and safety.
"""


# ---------------------------------------------------------
# create_documents()
# Converts raw text into LangChain Document objects.
#
# Internally:
# 1. Text is broken into sentences
# 2. Each sentence is converted into embeddingss
# 3. Semantic similarity between sentences is calculated
# 4. When a large meaning change is detected → new chunk created
# ---------------------------------------------------------
docs = text_splitter.create_documents([sample])


# Print how many semantic chunks were created
print(len(docs))


# Print the generated Document chunks
# Each item contains:
#   - page_content → actual chunk text
#   - metadata → additional info (if any)
print(docs)
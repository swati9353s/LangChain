from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import os

# -------------------------------------------------
# Step 1: Load environment variables from .env file
# This will load:
#   LLM_API_KEY  -> your API key
#   LLM_BASE_URL -> your custom LLM server endpoint
# -------------------------------------------------
load_dotenv()

# -------------------------------------------------
# Step 2: Initialize the embedding model
# We are NOT calling OpenAI directly.
# We are using our own LLM server which hosts:
#   intfloat/multilingual-e5-large embedding model
# -------------------------------------------------
embedding = OpenAIEmbeddings(
    model='intfloat/multilingual-e5-large',
    api_key=os.getenv('LLM_API_KEY'),      # API key from .env
    base_url=os.getenv('LLM_BASE_URL')     # Custom server URL
)

# -------------------------------------------------
# Step 3: List of documents we want to search from
# Each sentence is treated as one document
# -------------------------------------------------
documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

# -------------------------------------------------
# Step 4: User query (what the user wants to search)
# -------------------------------------------------
query = "Tell me about virat kohli"

# -------------------------------------------------
# Step 5: Convert all documents into embedding vectors
# This returns a list of vectors (one vector per document)
# -------------------------------------------------
doc_embeddings = embedding.embed_documents(documents)

# -------------------------------------------------
# Step 6: Convert user query into embedding vector
# This returns a single vector for the query
# -------------------------------------------------
query_embedding = embedding.embed_query(query)

# -------------------------------------------------
# Step 7: Calculate cosine similarity
# We compare:
#   query vector  VS  each document vector
#
# Result:
#   A list of similarity scores (one score per document)
# Higher score = more similar meaning
# -------------------------------------------------
# cosine_similarity returns a 2D array (matrix) even if we pass only one query embedding.
# Since we have only one query, the result will look like: [[score1, score2, score3, ...]]
# We use [0] to extract the first row and get a simple 1D list of similarity scores for each document.
scores = cosine_similarity([query_embedding], doc_embeddings)[0]

# -------------------------------------------------
# Step 8: Find the document with highest similarity
# enumerate(scores) -> gives (index, score)
# sorted by score
# [-1] picks the highest score (best match)
# -------------------------------------------------
# Enumerate the scores to get (index, score) pairs
# Sort the pairs based on score in ascending order
# sorted(..., key=lambda x: x[1]) tells Python to sort these pairs
# based on the second value of each tuple (which is the similarity score)
index, score = sorted(
    list(enumerate(scores)),
    key=lambda x: x[1]
)[-1]

# -------------------------------------------------
# Step 9: Print the most relevant document
# and its similarity score
# -------------------------------------------------
print("Most relevant document:")
print(documents[index])

print("\nSimilarity score is:", score)

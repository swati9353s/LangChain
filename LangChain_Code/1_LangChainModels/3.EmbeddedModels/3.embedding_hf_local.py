from langchain_huggingface import HuggingFaceEmbeddings  
# HuggingFaceEmbeddings lets us use Hugging Face embedding models locally or via API

# Create an embedding model using a popular sentence-transformer model
embedding = HuggingFaceEmbeddings(
    model_name='sentence-transformers/all-MiniLM-L6-v2'
)

# Convert a single text (query) into an embedding vector
result = embedding.embed_query("Hello world")

# -----------------------------
# For embedding multiple documents instead of a single query:
#
# documents = [
#     "Delhi is the capital of India",
#     "Kolkata is the capital of West Bengal"
# ]
#
# result = embedding.embed_documents(documents)
# print(result)
# -----------------------------

# Print the embedding vector for "Hello world"

print(str(result))

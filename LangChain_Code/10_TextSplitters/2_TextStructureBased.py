# -------------------- Import --------------------

# RecursiveCharacterTextSplitter is used to break large text
# into smaller chunks that LLMs can process safely.
from langchain_text_splitters import RecursiveCharacterTextSplitter


# -------------------- Sample Text --------------------

# Large paragraph which we want to divide into smaller parts
text = """
Space exploration has led to incredible scientific discoveries. From landing on the Moon to exploring Mars, humanity continues to push the boundaries of what’s possible beyond our planet.

These missions have not only expanded our knowledge of the universe but have also contributed to advancements in technology here on Earth. Satellite communications, GPS, and even certain medical imaging techniques trace their roots back to innovations driven by space programs.
"""


# -------------------- Initialize Text Splitter --------------------
# splits the paragraph first then sentence and then the words and characters based on the chunk size
splitter = RecursiveCharacterTextSplitter(
    
    # Maximum size of each chunk (in characters)
    chunk_size=2,
    
    # Number of overlapping characters between consecutive chunks
    # Overlap helps maintain context between chunks
    chunk_overlap=0,
)


# -------------------- Split the Text --------------------

# split_text() divides the large text into smaller chunks
# Returns a LIST of strings
chunks = splitter.split_text(text)


# -------------------- Output --------------------

# Number of chunks created
print(len(chunks))

# Actual chunks after splitting
print(chunks)

#Limitation is - only useful for text based input
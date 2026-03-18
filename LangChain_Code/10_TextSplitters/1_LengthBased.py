# -------------------- Imports --------------------

# CharacterTextSplitter splits documents based on characters
from langchain_text_splitters import CharacterTextSplitter

# PyPDFLoader reads PDF files and converts pages into Document objects
from langchain_community.document_loaders import PyPDFLoader


# -------------------- Load PDF --------------------

# Create loader for the PDF file
loader = PyPDFLoader('dl-curriculum.pdf')

# load() reads the PDF page-by-page
# Each page becomes a Document object
docs = loader.load()


# -------------------- Initialize Text Splitter --------------------

splitter = CharacterTextSplitter(
    
    # Maximum characters allowed in one chunk
    chunk_size=200,
    
    # No overlap between chunks
    chunk_overlap=0,
    
    # '' means split purely by character count
    # (no sentence or paragraph awareness)
    separator=''
)


# -------------------- Split Documents --------------------

# split_documents() takes Document objects as input
# and returns NEW smaller Document chunks
result = splitter.split_documents(docs)


# -------------------- Output --------------------

# Print content of second chunk created after splitting
print(result[1].page_content)

#problem with chunking happens abruptly
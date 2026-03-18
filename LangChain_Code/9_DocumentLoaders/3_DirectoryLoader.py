# -------------------- Imports --------------------

# DirectoryLoader loads multiple files from a folder
# PyPDFLoader is used internally to read each PDF file
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader


# -------------------- Create Directory Loader --------------------

loader = DirectoryLoader(
    path='books',        # Folder containing PDF files
    glob='*.pdf',        # Select only files ending with .pdf
    loader_cls=PyPDFLoader  # Use PyPDFLoader to read each file
)


# -------------------- Lazy Loading --------------------

# lazy_load() does NOT load all documents at once.
# Instead, it returns a generator (iterator).
# Documents are loaded one-by-one only when needed.
docs = loader.lazy_load()


# -------------------- Iterate Through Documents --------------------

# Each iteration loads the next page/document dynamically
for document in docs:
    
    # metadata contains information such as:
    # - source file path
    # - page number
    print(document.metadata)
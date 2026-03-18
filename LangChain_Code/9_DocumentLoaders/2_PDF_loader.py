# -------------------- Import --------------------

# PyPDFLoader is used to read PDF files
# and convert each page into a LangChain Document object
from langchain_community.document_loaders import PyPDFLoader


# -------------------- Load PDF --------------------

# Create loader object and provide PDF file path
loader = PyPDFLoader('dl-curriculum.pdf')


# load() reads the PDF and splits it page-wise
# Each page becomes one Document object
docs = loader.load()


# -------------------- Inspect Loaded Data --------------------

# Total number of pages (Documents) extracted from PDF
print(len(docs))


# Print text content of first page
# page_content contains actual extracted text
print(docs[0].page_content)


# Print metadata of second page
# metadata usually contains:
#   - source (file name)
#   - page number
print(docs[1].metadata)
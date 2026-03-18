# -------------------- Import --------------------

# CSVLoader reads a CSV file and converts each row
# into a LangChain Document object
from langchain_community.document_loaders import CSVLoader


# -------------------- Create Loader --------------------

# Provide path of CSV file
loader = CSVLoader(file_path='Social_Network_Ads.csv')


# -------------------- Load CSV --------------------

# load() reads the CSV file
# Each ROW in the CSV becomes one Document object
docs = loader.load()


# -------------------- Inspect Data --------------------

# Total number of rows (documents) in CSV
print(len(docs))


# Print second row as a Document object
print(docs[1])
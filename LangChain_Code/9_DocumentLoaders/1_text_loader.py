# -------------------- Imports --------------------

# Loads text files and converts them into LangChain Document objects
from langchain_community.document_loaders import TextLoader

# OpenAI chat model wrapper (LLM)
from langchain_openai import ChatOpenAI

# Converts model output into plain string text
from langchain_core.output_parsers import StrOutputParser

# Used to create dynamic prompts with variables
from langchain_core.prompts import PromptTemplate

# Loads environment variables (like OPENAI_API_KEY) from .env file
from dotenv import load_dotenv
import os


# -------------------- Load Environment Variables --------------------
# Reads API key from .env file so model can authenticate
load_dotenv()


# -------------------- Initialize Model --------------------
# Creates an LLM instance (ChatGPT model)
model = ChatOpenAI(
    model='mistralai/Mistral-Small-3.1-24B-Instruct-2503',
    api_key=os.getenv("LLM_API_KEY"),     # API key from .env
    base_url=os.getenv("LLM_BASE_URL")    # Custom provider endpoint
)


# -------------------- Create Prompt Template --------------------
# {poem} is a placeholder that will be replaced at runtime
prompt = PromptTemplate(
    template='Write a summary for the following poem - \n {poem}',
    input_variables=['poem']
)


# -------------------- Output Parser --------------------
# LLM returns structured message objects.
# This parser extracts only the text response.
parser = StrOutputParser()


# -------------------- Load Text File --------------------
# Reads cricket.txt file and converts it into Document objects
# Each Document contains:
#   - page_content → actual text
#   - metadata → file information
loader = TextLoader('cricket.txt', encoding='utf-8')


# load() reads file and returns a LIST of Documents
docs = loader.load()


# -------------------- Inspect Loaded Data --------------------

# Type of docs (should be list)
print(type(docs))

# Number of documents created from the file
print(len(docs))

# Actual text content of first document
print(docs[0].page_content)

# Metadata (file path, source info, etc.)
print(docs[0].metadata)


# -------------------- Create LLM Chain --------------------
# LCEL pipeline:
# Prompt → Model → Parser
chain = prompt | model | parser


# -------------------- Execute Chain --------------------
# Pass poem text into prompt variable {poem}
# Model generates summary
# Parser converts output to string
print(chain.invoke({'poem': docs[0].page_content}))
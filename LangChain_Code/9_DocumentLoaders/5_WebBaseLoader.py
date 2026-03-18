# -------------------- Imports --------------------

# Loads webpage content and converts it into Document objects
from langchain_community.document_loaders import WebBaseLoader

# OpenAI chat model (LLM)
from langchain_openai import ChatOpenAI

# Converts LLM output into plain string text
from langchain_core.output_parsers import StrOutputParser

# Used to create prompts with dynamic variables
from langchain_core.prompts import PromptTemplate

# Loads API keys from .env file
from dotenv import load_dotenv

import os


# -------------------- Load Environment Variables --------------------
# Reads OPENAI_API_KEY from .env
load_dotenv()


# -------------------- Initialize Model --------------------
# Creates ChatGPT model instance
model = ChatOpenAI(
    model='mistralai/Mistral-Small-3.1-24B-Instruct-2503',
    api_key=os.getenv("LLM_API_KEY"),     # API key from .env
    base_url=os.getenv("LLM_BASE_URL")    # Custom provider endpoint
)


# -------------------- Prompt Template --------------------
# The LLM will answer a question using provided webpage text
# {question} and {text} will be filled at runtime
prompt = PromptTemplate(
    template='Answer the following question \n {question} from the following text - \n {text}',
    input_variables=['question', 'text']
)


# -------------------- Output Parser --------------------
# Converts model response object → simple string
parser = StrOutputParser()


# -------------------- Web Loader --------------------
# URL of webpage to scrape and load
url = 'https://www.flipkart.com/apple-macbook-air-m2-16-gb-256-gb-ssd-macos-sequoia-mc7x4hn-a/p/itmdc5308fa78421'

# WebBaseLoader fetches webpage HTML and extracts readable text
# It will use the bs4 internally
loader = WebBaseLoader(url)


# load() returns a list of Document objects
# Each Document contains:
#   - page_content → webpage text
#   - metadata → source URL info
docs = loader.load()


# -------------------- Create Chain --------------------
# Flow:
# Prompt → Model → Parser
chain = prompt | model | parser


# -------------------- Execute Chain --------------------
# Pass:
#   question → what we want to ask
#   text → webpage content
#
# LLM reads webpage text and answers the question
print(
    chain.invoke({
        'question': 'What is the product that we are talking about?',
        'text': docs[0].page_content
    })
)
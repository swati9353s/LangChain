# Import PromptTemplate to create dynamic prompts
from langchain_core.prompts import PromptTemplate

# Import ChatOpenAI to communicate with the LLM
from langchain_openai import ChatOpenAI

# Used to load environment variables from .env file
from dotenv import load_dotenv

# os module helps access environment variables
import os


# ---------------------------------------------------------
# Load API key and base URL from .env file
# ---------------------------------------------------------
load_dotenv()


# ---------------------------------------------------------
# Initialize the LLM model
# model      -> Name of the LLM
# api_key    -> Authentication key
# base_url   -> Endpoint of model provider
# ---------------------------------------------------------
model = ChatOpenAI(
    model='mistralai/Mistral-Small-3.1-24B-Instruct-2503',
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL")
)


# ---------------------------------------------------------
# Create a PromptTemplate
# {name} is a placeholder which will be filled at runtime
# input_variables tells LangChain which variables are required
# ---------------------------------------------------------
template = PromptTemplate(
    template="Greet this person in 5 languages, the name of the person is {name}",
    input_variables=['name']
)


# ------------------------------------------

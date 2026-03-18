# -------------------- IMPORTS --------------------

# ChatOpenAI:
# LangChain wrapper that allows us to talk to any
# OpenAI-compatible LLM (OpenAI, OpenRouter, Together, etc.)
from langchain_openai import ChatOpenAI

# load_dotenv:
# Loads environment variables from a .env file into Python
# so we can safely store API keys outside the code.
from dotenv import load_dotenv

# PromptTemplate:
# Used to create reusable prompts with variables.
# Example: "Tell me about {topic}"
from langchain_core.prompts import PromptTemplate

# StrOutputParser:
# Converts model output into a plain Python string.
# (LLMs return structured message objects internally)
from langchain_core.output_parsers import StrOutputParser

# os:
# Used to read environment variables like API keys
import os


# -------------------- LOAD ENV VARIABLES --------------------

# Reads .env file and makes variables available via os.getenv()
load_dotenv(r"C:\Users\smi68\Desktop\My_Learning\Artificial-Intelligence\LangChain\LangChain_Code\.env")


# -------------------- PROMPT TEMPLATE --------------------

# PromptTemplate defines HOW we talk to the LLM.
# It is like a reusable prompt function.
prompt = PromptTemplate(

    # template:
    # The actual instruction sent to the LLM.
    # {topic} is a placeholder variable.
    template='Generate 5 interesting facts about {topic}',

    # input_variables:
    # List of variables that must be supplied when running the chain.
    input_variables=['topic']
)


# -------------------- MODEL (LLM) SETUP --------------------

# ChatOpenAI creates a connection to the language model.
model = ChatOpenAI(

    # model:
    # Name of the LLM provided by your API provider.
    # Must match exactly what provider supports.
    model='mistralai/mistral-small-2506',

    # api_key:
    # Authentication key stored securely in .env
    api_key=os.getenv("LLM_API_KEY"),

    # base_url:
    # Endpoint of OpenAI-compatible provider.
    # Example: OpenRouter / Together / Local server
    base_url=os.getenv("LLM_BASE_URL")
)


# -------------------- OUTPUT PARSER --------------------

# LLM responses are not simple strings.
# They are AIMessage objects containing metadata.
# This parser extracts only the text content.
parser = StrOutputParser()


# -------------------- BUILDING THE CHAIN --------------------

# A "Chain" = pipeline of steps executed sequentially.
#
# LCEL (LangChain Expression Language) operator "|"
# connects components together.
#
# Flow:
# User Input -> PromptTemplate -> Model -> Parser -> Final Output

chain = prompt | model | parser


# -------------------- EXECUTE THE CHAIN --------------------

# invoke():
# Runs the chain once with given inputs.
#
# We must pass a dictionary matching input_variables.
result = chain.invoke({
    'topic': 'cricket'
})

# Print final processed output
print(result)


# -------------------- VISUALIZE CHAIN STRUCTURE --------------------

# get_graph():
# Builds an internal execution graph of the chain.
#
# print_ascii():
# Displays pipeline structure in terminal.
chain.get_graph().print_ascii()
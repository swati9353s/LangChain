# ---------------------------------------------------------
# Import ChatPromptTemplate for dynamic chat prompts
# ---------------------------------------------------------
from langchain_core.prompts import ChatPromptTemplate

# ChatOpenAI -> interface to interact with LLM
from langchain_openai import ChatOpenAI

# Load environment variables
from dotenv import load_dotenv
import os


# ---------------------------------------------------------
# Load API configuration
# ---------------------------------------------------------
load_dotenv()


# ---------------------------------------------------------
# Initialize the LLM model
# ---------------------------------------------------------
model = ChatOpenAI(
    model='mistralai/Mistral-Small-3.1-24B-Instruct-2503',
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL")
)


# ---------------------------------------------------------
# Create ChatPromptTemplate
#
# IMPORTANT:
# In ChatPromptTemplate we define messages using tuples:
# (role, template_string)
#
# Roles can be:
#   "system", "human", "ai"
#
# Placeholders like {domain} and {topic}
# will be filled dynamically at runtime.
# ---------------------------------------------------------
template = ChatPromptTemplate([
    ("system", "You are an AI expert in the {domain}."),
    ("human", "Explain the {topic}.")
])


# ---------------------------------------------------------
# Fill placeholders dynamically
# invoke() returns a list of formatted messages
# ---------------------------------------------------------
prompt = template.invoke({
    'domain': 'AI',
    'topic': 'LangChain'
})


# ---------------------------------------------------------
# Print generated messages (chat format)
# ---------------------------------------------------------
print(prompt)

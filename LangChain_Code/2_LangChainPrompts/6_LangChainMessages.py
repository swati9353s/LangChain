# ---------------------------------------------------------
# Import message classes used in conversational LLMs
# ---------------------------------------------------------
# SystemMessage -> defines AI behaviour/instructions
# HumanMessage  -> represents user input
# AIMessage     -> represents LLM response
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# ChatOpenAI -> interface to interact with the LLM
from langchain_openai import ChatOpenAI

# Load environment variables from .env file
from dotenv import load_dotenv

# Used to access environment variables
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
# Define conversation messages
# LLM receives conversation as a LIST of messages
# ---------------------------------------------------------
messages = [

    # System message sets assistant behaviour
    SystemMessage(content="You are a useful AI assistant"),

    # First user query
    HumanMessage(content="Tell me about LangChain")
]


# ---------------------------------------------------------
# Send messages to the model
# The entire list acts as conversation context
# ---------------------------------------------------------
result = model.invoke(messages)


# ---------------------------------------------------------
# Store AI response into conversation history
# This is important for maintaining context
# ---------------------------------------------------------
messages.append(AIMessage(content=result.content))


# ---------------------------------------------------------
# Print complete message history
# (Useful for understanding internal structure)
# ---------------------------------------------------------
print(messages)

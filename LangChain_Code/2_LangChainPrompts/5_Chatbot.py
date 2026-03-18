# ---------------------------------------------------------
# Import required modules
# ---------------------------------------------------------

# ChatOpenAI -> used to communicate with LLM
from langchain_openai import ChatOpenAI

# Loads environment variables from .env file
from dotenv import load_dotenv

# Used to access environment variables
import os

# Message classes used for conversational context
# SystemMessage -> instructions for AI behavior
# HumanMessage  -> user input
# AIMessage     -> model response
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


# ---------------------------------------------------------
# Load API key and configuration
# ---------------------------------------------------------
load_dotenv()


# ---------------------------------------------------------
# Initialize the LLM
# ---------------------------------------------------------
model = ChatOpenAI(
    model='mistralai/Mistral-Small-3.1-24B-Instruct-2503',
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL")
)


# ---------------------------------------------------------
# Create chat history
# LLMs are stateless by default, so we manually maintain
# conversation history and send it every time.
# ---------------------------------------------------------
chat_history = [
    # System message defines AI behaviour/personality
    SystemMessage(content="You are a useful AI assistant")
]


# ---------------------------------------------------------
# Continuous chatbot loop
# ---------------------------------------------------------
while True:

    # Take input from user
    user_input = input('You: ')

    # Exit condition
    if user_input == 'exit':
        break

    # Add user message to chat history
    chat_history.append(HumanMessage(content=user_input))

    # Send entire conversation history to model
    # This provides context awareness
    result = model.invoke(chat_history)

    # Store AI response in chat history
    chat_history.append(AIMessage(content=result.content))

    # Print AI response
    print('AI:', result.content)


# ---------------------------------------------------------
# Print full conversation history (for debugging/learning)
# ---------------------------------------------------------
print(chat_history)

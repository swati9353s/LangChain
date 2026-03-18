# ---------------------------------------------------------
# Import required classes
# ---------------------------------------------------------
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

# Message classes (required for chat history)
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage


# ---------------------------------------------------------
# Load environment variables (.env)
# ---------------------------------------------------------
load_dotenv()


# ---------------------------------------------------------
# Initialize LLM model
# ---------------------------------------------------------
model = ChatOpenAI(
    model='mistralai/Mistral-Small-3.1-24B-Instruct-2503',
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL")
)


# ---------------------------------------------------------
# Create ChatPromptTemplate
#
# MessagesPlaceholder allows dynamic insertion
# of past conversation history at runtime.
# ---------------------------------------------------------
template = ChatPromptTemplate([
    ("system", "You are a useful AI assistant"),

    # Placeholder where previous messages will be inserted
    MessagesPlaceholder(variable_name='chat_history'),

    # Current user query
    ("human", "{query}")
])


# ---------------------------------------------------------
# Chat history container
# IMPORTANT: Must contain Message objects
# ---------------------------------------------------------
chat_history = []


# ---------------------------------------------------------
# Load chat history from file
# NOTE:
# readlines() returns plain strings ❌
# We must convert them into HumanMessage / AIMessage ✅
# ---------------------------------------------------------
with open('chat_history.txt') as f:
    for line in f:
        line = line.strip()

        # Example format assumption:
        # Human: hello
        # AI: hi there
        if line.startswith("Human:"):
            chat_history.append(
                HumanMessage(content=line.replace("Human:", "").strip())
            )

        elif line.startswith("AI:"):
            chat_history.append(
                AIMessage(content=line.replace("AI:", "").strip())
            )


# Debug: check loaded structured messages
print(chat_history)


# ---------------------------------------------------------
# Create final prompt by injecting:
# 1. system message
# 2. past chat history
# 3. current query
# ---------------------------------------------------------
prompt = template.invoke({
    'chat_history': chat_history,
    'query': 'where is my refund'
})


# Print final structured prompt
print(prompt)

# Import ChatOpenAI wrapper from LangChain
# This allows us to interact with an LLM using chat-based interface
from langchain_openai import ChatOpenAI

# Used to load environment variables from .env file
from dotenv import load_dotenv

# Streamlit is used to build the web UI
import streamlit as st

# os module helps access environment variables
import os


# ---------------------------------------------------------
# Load environment variables from .env file
# (API key and Base URL will be read from here)
# ---------------------------------------------------------
load_dotenv()


# ---------------------------------------------------------
# Initialize the LLM model
# model      -> Name of the model we want to use
# api_key    -> Authentication key (kept secure in .env)
# base_url   -> Endpoint of the LLM provider
# ---------------------------------------------------------
model = ChatOpenAI(
    model='mistralai/Mistral-Small-3.1-24B-Instruct-2503',
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL")
)


# ---------------------------------------------------------
# Create a header/title in Streamlit UI
# ---------------------------------------------------------
st.header("Research Tool")


# ---------------------------------------------------------
# Create a text input box where user enters prompt
# The entered text is stored in 'user_input'
# ---------------------------------------------------------
user_input = st.text_input("Enter your prompt:")


# ---------------------------------------------------------
# When user clicks the "Summarize" button:
# 1. Send input to LLM
# 2. Get response
# 3. Display output on UI
# ---------------------------------------------------------
if st.button("Summarize"):

    # Send user prompt to the model
    # invoke() sends the request and returns AIMessage object
    result = model.invoke(user_input)

    # result.content contains only the generated text response
    st.write(result.content)

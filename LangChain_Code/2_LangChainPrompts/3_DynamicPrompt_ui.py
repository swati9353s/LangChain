# ---------------------------------------------------------
# Import required libraries
# ---------------------------------------------------------

# ChatOpenAI -> Used to interact with LLM models
from langchain_openai import ChatOpenAI

# Loads environment variables from .env file
from dotenv import load_dotenv

# Streamlit -> Used to build web UI
import streamlit as st

# Used to access environment variables
import os

# PromptTemplate -> create dynamic prompts
# load_prompt -> load prompt from external JSON file
from langchain_core.prompts import PromptTemplate, load_prompt


# ---------------------------------------------------------
# Load API keys and configuration from .env file
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
# Streamlit UI Section
# ---------------------------------------------------------

# App title
st.header("Research Tool")


# Dropdown to select research paper
paper_input = st.selectbox(
    "Select Research Paper Name",
    [
        "Attention Is All You Need",
        "BERT: Pre-training of Deep Bidirectional Transformers",
        "GPT-3: Language Models are Few-Shot Learners",
        "Diffusion Models Beat GANs on Image Synthesis"
    ]
)

# Dropdown to select explanation style
style_input = st.selectbox(
    "Select Explanation Style",
    ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"]
)

# Dropdown to select explanation length
length_input = st.selectbox(
    "Select Explanation Length",
    ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"]
)


# ---------------------------------------------------------
# Prompt Handling
# In real-world applications:
# - Prompts are large
# - We separate prompt logic from application code
# - Prompt is stored externally (JSON/YAML)
# ---------------------------------------------------------

# Load prompt template from JSON file
template = load_prompt("promptTemplate.json")


# ---------------------------------------------------------
# NOTE:
# Earlier approach required two invoke() calls:
# 1. template.invoke() -> create formatted prompt
# 2. model.invoke() -> send prompt to LLM
#
# LangChain Expression Language (LCEL)
# allows chaining using the | operator
# ---------------------------------------------------------


# ---------------------------------------------------------
# When user clicks Summarize button
# ---------------------------------------------------------
if st.button("Summarize"):

    # Create a chain:
    # PromptTemplate output automatically flows into model input
    chain = template | model

    # Invoke chain with dynamic inputs
    result = chain.invoke({
        "paper": paper_input,
        "style": style_input,
        "length": length_input
    })

    # Display generated response
    st.write(result.content)

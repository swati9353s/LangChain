# Import required libraries and classes
from langchain_openai import ChatOpenAI  # OpenAI LLM wrapper for LangChain
from langchain_core.prompts import PromptTemplate  # For defining prompt templates
from langchain_core.output_parsers import StrOutputParser  # To parse LLM output as string
from dotenv import load_dotenv  # Load environment variables (like API keys)
from langchain_core.runnables import RunnableSequence  # To chain multiple runnables
import os

# Load environment variables from a .env file
load_dotenv()

# ------------------- Prompt Templates -------------------
# First prompt: asks the LLM to generate a joke about a given topic
prompt1 = PromptTemplate(
    template='Write a joke about {topic}',  # The template string with a placeholder
    input_variables=['topic']  # List of variables to be replaced in the template
)

# Second prompt: asks the LLM to explain a joke
prompt2 = PromptTemplate(
    template='Explain the following joke - {text}',  # Template with a placeholder for the joke
    input_variables=['text']
)

# ------------------- LLM Model -------------------
# Initialize the OpenAI LLM model
model = ChatOpenAI(
    model='mistralai/Mistral-Small-3.1-24B-Instruct-2503',
    api_key=os.getenv("LLM_API_KEY"),     # API key from .env
    base_url=os.getenv("LLM_BASE_URL")    # Custom provider endpoint
)

# ------------------- Output Parser -------------------
# Parser to convert LLM output into a simple string
parser = StrOutputParser()

# ------------------- RunnableSequence -------------------
# Chain multiple runnables in sequence:
# 1. prompt1 → model → parser → produces a joke
# 2. prompt2 → model → parser → explains the joke
# RunnableSequence ensures the output of one step is passed as input to the next
chain = RunnableSequence(
    prompt1, model, parser,   # Step 1: generate joke
    prompt2, model, parser    # Step 2: explain joke
)

# ------------------- Run the Chain -------------------
# Invoke the chain with input variables
# {'topic':'AI'} replaces {topic} in prompt1
# The chain automatically passes output from prompt1 -> model -> parser -> prompt2 -> model -> parser
print(chain.invoke({'topic':'AI'}))  # Final output: explanation of the AI joke
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv()

model = ChatOpenAI(
    model='mistralai/Mistral-Small-3.1-24B-Instruct-2503',
    api_key=os.getenv("LLM_API_KEY"),     # API key from .env
    base_url=os.getenv("LLM_BASE_URL")    # Custom provider endpoint
)

# ---------------- PROMPT 1 ----------------
# This prompt is responsible for generating a FULL detailed report.
# We use PromptTemplate so that the prompt becomes reusable and dynamic.
# {topic} acts as a placeholder which will be replaced at runtime.
prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)

# ---------------- PROMPT 2 ----------------
# This prompt takes the output produced by the first prompt
# and converts it into a summarized version.
# The {text} variable will automatically receive the previous step output.
prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary from the following text \n {text}',
    input_variables=['text']
)

# ---------------- OUTPUT PARSER ----------------
# StrOutputParser ensures that the LLM response is converted into
# a clean Python string.
# LLMs often return structured message objects; this parser extracts
# only the text content so it can be passed to the next step in the chain.
parser = StrOutputParser()

# ---------------- CHAIN CREATION ----------------
# The "|" operator connects multiple components into a pipeline.
#
# Flow of execution:
# 1. prompt1 -> creates formatted prompt using topic input
# 2. model   -> generates detailed report
# 3. parser  -> converts model response into plain string
# 4. prompt2 -> injects that string into summary prompt
# 5. model   -> generates summary
# 6. parser  -> again converts final output into string
#
# Why use chains?
# - Automates multi-step LLM workflows
# - Removes manual handling of intermediate outputs
# - Makes pipelines readable and modular
# - Easy to extend (add validation, memory, tools later)
chain = prompt1 | model | parser | prompt2 | model | parser

# ---------------- EXECUTION ----------------
# invoke() runs the entire chain.
# Input dictionary values replace variables defined in PromptTemplate.
# Here, {topic} gets replaced with "Unemployment in India".
result = chain.invoke({'topic': 'Unemployment in India'})

# Prints the final summarized output returned by the chain.
print(result)

# ---------------- CHAIN VISUALIZATION ----------------
# get_graph() builds an internal execution graph of the chain.
# print_ascii() displays a text-based diagram showing how data
# flows between prompts, model, and parsers.
# Useful for debugging and understanding pipeline structure.
chain.get_graph().print_ascii()
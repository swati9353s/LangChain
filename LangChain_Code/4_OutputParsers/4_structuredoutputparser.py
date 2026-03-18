from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
import os
# It is not getting imported, may be some version issue
from langchain.output_parsers import ResponseSchema, StructuredOutputParser

load_dotenv()  # Reads variables from .env file and makes them available using os.getenv()

model = ChatOpenAI(
    model='mistralai/Mistral-Small-3.1-24B-Instruct-2503',
    api_key=os.getenv("LLM_API_KEY"),     # Retrieves API key securely from environment variables
    base_url=os.getenv("LLM_BASE_URL")    # Reads custom model endpoint if configured
)

# ---------------- RESPONSE STRUCTURE ----------------
# ResponseSchema:
# Defines what fields must appear in the final LLM output.
# Each schema entry tells the model:
#   - the key name expected in output
#   - what information should be placed inside it.
schema = [
    ResponseSchema(name='fact_1', description='Fact 1 about the topic'),
    ResponseSchema(name='fact_2', description='Fact 2 about the topic'),
    ResponseSchema(name='fact_3', description='Fact 3 about the topic'),
]

# StructuredOutputParser.from_response_schemas():
# Builds a parser using the defined schema list.
# The parser later:
#   - forces the model to respond in a structured format
#   - converts raw text output into a Python dictionary.
parser = StructuredOutputParser.from_response_schemas(schema)

# get_format_instructions():
# Generates formatting rules automatically (usually JSON format instructions).
# These instructions are inserted into the prompt so the LLM knows
# exactly how to structure its response.
# This reduces hallucinated or free-text outputs.

# ---------------- PROMPT TEMPLATE ----------------
# PromptTemplate dynamically inserts runtime values.
# input_variables:
#   Values that must be supplied during invocation.
# partial_variables:
#   Values automatically injected every time (no need to pass manually).
template = PromptTemplate(
    template='Give 3 fact about {topic} \n {format_instruction}',
    input_variables=['topic'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

# ---------------- CHAIN PIPELINE ----------------
# "|" operator creates a sequential execution pipeline.
#
# Flow:
# 1. Template formats the prompt using topic input.
# 2. Model generates response following format instructions.
# 3. StructuredOutputParser validates and converts output
#    into a structured Python dictionary.
#
# Benefit:
# - No manual JSON parsing
# - Predictable keys in output
# - Easy downstream processing in applications
chain = template | model | parser

# ---------------- EXECUTION ----------------
# invoke():
# Executes the entire chain once.
# Replaces {topic} with provided value and passes data step-by-step.
# Returns a dictionary like:
# {'fact_1': '...', 'fact_2': '...', 'fact_3': '...'}
result = chain.invoke({'topic': 'black hole'})

# Prints structured dictionary output.
print(result)
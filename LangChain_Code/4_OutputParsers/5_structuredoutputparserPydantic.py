from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
import os
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()  # Loads environment variables (.env file) into the program so API keys and URLs can be accessed securely.

model = ChatOpenAI(
    model='mistralai/Mistral-Small-3.1-24B-Instruct-2503',
    api_key=os.getenv("LLM_API_KEY"),     # Fetches API key from environment variables instead of hard-coding it.
    base_url=os.getenv("LLM_BASE_URL")    # Reads custom endpoint if provided.
)

# ---------------- OUTPUT STRUCTURE DEFINITION ----------------
# BaseModel:
# Provides automatic data validation and structured data handling.
# Any output parsed into this model must follow the defined field types.

class Person(BaseModel):

    # Field():
    # Adds metadata and validation rules for each attribute.
    # description → helps the LLM understand what value to generate.
    name: str = Field(description='Name of the person')

    # gt=18:
    # Validation rule meaning "greater than 18".
    # If the model returns age <= 18, parsing will fail with validation error.
    age: int = Field(gt=18, description='Age of the person')

    city: str = Field(description='Name of the city the person belongs to')


# ---------------- OUTPUT PARSER ----------------
# PydanticOutputParser:
# Converts raw LLM text into a structured Python object.
# It also validates the output using the rules defined in Person model.
parser = PydanticOutputParser(pydantic_object=Person)

# get_format_instructions():
# Generates automatic instructions telling the LLM
# exactly how the output should be formatted (JSON schema-like structure).
# This greatly reduces incorrect or unstructured responses.

# ---------------- PROMPT TEMPLATE ----------------
# PromptTemplate:
# Dynamically builds prompts by replacing variables at runtime.
# partial_variables:
# Injects fixed values automatically (here, formatting instructions),
# so the user does not need to pass them during invocation.

template = PromptTemplate(
    template='Generate the name, age and city of a fictional {place} person \n {format_instruction}',
    input_variables=['place'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

# ---------------- CHAIN PIPELINE ----------------
# "|" operator connects components into a sequential workflow.
#
# Execution flow:
# 1. Template formats the prompt using input values.
# 2. Model generates response based on formatted prompt.
# 3. Parser converts response into validated Person object.
#
# Benefit:
# - No manual parsing required
# - Automatic validation
# - Clean structured output ready for program use
chain = template | model | parser


# ---------------- EXECUTION ----------------
# invoke():
# Runs the entire pipeline once.
# Replaces {place} with provided value and passes data through all steps.
# Returns a fully validated Person object instead of plain text.
final_result = chain.invoke({'place': 'sri lankan'})

# Printing the structured result.
# Output behaves like a normal Python object (final_result.name, etc.).
print(final_result)
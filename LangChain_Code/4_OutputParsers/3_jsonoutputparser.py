from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
import os
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()  # Loads environment variables so API keys and endpoints can be accessed securely.

model = ChatOpenAI(
    model='mistralai/Mistral-Small-3.1-24B-Instruct-2503',
    api_key=os.getenv("LLM_API_KEY"),     # Reads API key from environment instead of hardcoding.
    base_url=os.getenv("LLM_BASE_URL")    # Reads custom endpoint if provided.
)

# ---------------- JSON OUTPUT PARSER ----------------
# JsonOutputParser:
# Ensures the LLM response follows JSON format.
# It later converts the generated JSON text into a Python dictionary.
outputparser = JsonOutputParser()

# get_format_instructions():
# Automatically generates instructions telling the LLM
# how the response should be structured (valid JSON format).
# Adding these instructions reduces invalid or free-text responses.

# ---------------- PROMPT TEMPLATE ----------------
# PromptTemplate builds the final prompt dynamically.
# input_variables = [] means no runtime inputs are required.
# partial_variables inject fixed values automatically
# (here, JSON formatting rules).
template1 = PromptTemplate(
    template='give me a name, age and city of a fictional person\n {format_instructions}',
    input_variables=[],
    partial_variables={
        'format_instructions': outputparser.get_format_instructions()
    }
)

# format():
# Replaces placeholders inside the template and produces
# the final prompt string that will be sent to the model.
prompt = template1.format()

# Displays the complete prompt including JSON instructions.
# Useful for debugging and understanding what the model receives.
print(prompt)

# invoke():
# Sends the prompt to the LLM and returns a response object.
# The response contains metadata + generated text content.
result = model.invoke(prompt)

# Shows raw model output before parsing.
# At this stage it is still plain text.
print(result)

# parse():
# Extracts JSON content from the model response
# and converts it into a Python dictionary.
# If JSON format is incorrect, parsing will raise an error.
final_result = outputparser.parse(result.content)

# Final structured output that can be accessed like:
# final_result["name"], final_result["age"], etc.
print(final_result)


# ---------------- OPTIONAL: USING CHAIN ----------------
# Instead of manually calling format(), invoke(), and parse(),
# a chain automates the entire workflow.
#
# Flow:
# Template → Model → JSON Parser
#
# Benefit:
# - Less boilerplate code
# - Automatic passing of outputs between steps
# - Cleaner and scalable pipeline design

# chain = template1 | model | outputparser
# result = chain.invoke({})   # Empty dict required since no input variables exist
# print(result)
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()  
# Loads environment variables from the .env file so API tokens
# or configuration values can be accessed securely.

# ---------------- HUGGING FACE ENDPOINT ----------------
# HuggingFaceEndpoint:
# Connects LangChain to a Hugging Face hosted model.
# repo_id specifies which model to use.
# task tells the endpoint what type of generation is expected.
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

# ChatHuggingFace:
# Wraps the base LLM into a chat-compatible interface.
# This allows the model to work with LangChain chat workflows
# and return structured response objects.
model = ChatHuggingFace(llm=llm)

# ---------------- FIRST PROMPT ----------------
# PromptTemplate dynamically creates prompts.
# {topic} will be replaced with actual input during execution.
# This prompt asks the model to generate detailed information.
template1 = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)

# ---------------- SECOND PROMPT ----------------
# This template receives the output of the first model call.
# {text} will contain the generated report which will be summarized.
template2 = PromptTemplate(
    template='Write a 5 line summary on the following text. \n {text}',
    input_variables=['text']
)

# invoke():
# Fills the template placeholders using provided values
# and returns the formatted prompt ready for the model.
prompt1 = template1.invoke({'topic': 'black hole'})

# model.invoke():
# Sends the formatted prompt to the LLM.
# Returns a response object containing generated text + metadata.
result = model.invoke(prompt1)

# The first model output (result.content) is inserted into
# the second prompt template as input text.
prompt2 = template2.invoke({'text': result.content})

# Second model call generates the summarized version
# based on the detailed report.
result1 = model.invoke(prompt2)

# result1.content:
# Extracts only the generated message text from the response object.
# This is the final summarized output.
print(result1.content)
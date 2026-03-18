# -------------------- Imports --------------------

from langchain_openai import ChatOpenAI          # OpenAI chat model wrapper
from langchain_core.prompts import PromptTemplate # Used to create dynamic prompts
from langchain_core.output_parsers import StrOutputParser # Converts model output → plain text
from dotenv import load_dotenv                   # Loads environment variables (.env file)
from langchain_core.runnables import (
    RunnableSequence,
    RunnableParallel,
    RunnablePassthrough,
    RunnableBranch,
    RunnableLambda
)
import os

# Load API key (OPENAI_API_KEY) from .env file
load_dotenv()


# -------------------- Prompt 1 --------------------
# Generates a detailed report based on a topic.
# {topic} will be replaced at runtime.

prompt1 = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)


# -------------------- Prompt 2 --------------------
# Used only when the generated report is too long.
# It summarizes the generated text.

prompt2 = PromptTemplate(
    template='Summarize the following text \n {text}',
    input_variables=['text']
)


# -------------------- LLM Model --------------------
# Initializes ChatGPT/OpenAI model
model = ChatOpenAI(
    model='mistralai/Mistral-Small-3.1-24B-Instruct-2503',
    api_key=os.getenv("LLM_API_KEY"),     # API key from .env
    base_url=os.getenv("LLM_BASE_URL")    # Custom provider endpoint
)

# -------------------- Output Parser --------------------
# Converts LLM output object into simple string text
parser = StrOutputParser()


# -------------------- Report Generation Chain --------------------
# Using LCEL (| operator)
#
# Flow:
# Prompt → LLM → Parser
#
# Input  : {'topic': 'Russia vs Ukraine'}
# Output : Large generated report (string)

report_gen_chain = prompt1 | model | parser


# -------------------- Branch Chain --------------------
# RunnableBranch works like an IF–ELSE condition.
#
# Condition:
# If generated text has more than 300 words → summarize it
# Else → return text as it is.

branch_chain = RunnableBranch(

    # CONDITION + TRUE PATH
    # lambda receives output text from previous chain
    # If condition is True → run summarization chain
    (lambda x: len(x.split()) > 300,
     prompt2 | model | parser),

    # ELSE CASE
    # If condition is False, pass original text unchanged
    RunnablePassthrough()
)


# -------------------- Final Chain --------------------
# Step 1: Generate report
# Step 2: Decide whether to summarize or not

final_chain = RunnableSequence(
    report_gen_chain,
    branch_chain
)


# -------------------- Execute Chain --------------------
# Input topic is provided here
# Output will be:
# - summarized report (if long)
# OR
# - original report (if short)

print(final_chain.invoke({'topic': 'Russia vs Ukraine'}))
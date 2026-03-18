# -------------------- IMPORTS --------------------

# OpenAI-compatible chat model wrapper
from langchain_openai import ChatOpenAI

# Loads API keys from .env file
from dotenv import load_dotenv

# Used to create reusable prompts with variables
from langchain_core.prompts import PromptTemplate

# Converts LLM output into plain string
from langchain_core.output_parsers import StrOutputParser

# Runnable components (NEW LOCATION in modern LangChain)
# RunnableParallel  -> run multiple chains together
# RunnableBranch    -> conditional routing (if/else logic)
# RunnableLambda    -> custom python function inside chain
from langchain_core.runnables import (
    RunnableParallel,
    RunnableBranch,
    RunnableLambda
)

# Parser that converts LLM output → Pydantic object
from langchain_core.output_parsers import PydanticOutputParser

# Used for structured output validation
from pydantic import BaseModel, Field

# Restricts allowed values
from typing import Literal

import os


# -------------------- LOAD ENV VARIABLES --------------------
load_dotenv()


# -------------------- MODEL --------------------

# Default ChatOpenAI uses OPENAI_API_KEY from .env
# ChatOpenAI creates a connection to the language model.
model = ChatOpenAI(

    # model:
    # Name of the LLM provided by your API provider.
    # Must match exactly what provider supports.
    model='mistralai/Mistral-Small-3.1-24B-Instruct-2503',

    # api_key:
    # Authentication key stored securely in .env
    api_key=os.getenv("LLM_API_KEY"),

    # base_url:
    # Endpoint of OpenAI-compatible provider.
    # Example: OpenRouter / Together / Local server
    base_url=os.getenv("LLM_BASE_URL")
)



# -------------------- BASIC STRING PARSER --------------------

# Extracts only text from AIMessage response
parser = StrOutputParser()


# -------------------- STRUCTURED OUTPUT SCHEMA --------------------

# We define the structure we EXPECT from LLM output
class Feedback(BaseModel):

    # Literal ensures ONLY these values are valid
    sentiment: Literal['positive', 'negative'] = Field(
        description='Give the sentiment of the feedback'
    )


# Converts LLM output into Feedback object
parser2 = PydanticOutputParser(pydantic_object=Feedback)


# -------------------- CLASSIFICATION PROMPT --------------------

prompt1 = PromptTemplate(

    # LLM is instructed to classify sentiment
    template='''
    Classify the sentiment of the following feedback text
    into positive or negative.

    {feedback}

    {format_instruction}
    ''',

    # Input provided at runtime
    input_variables=['feedback'],

    # Automatically inject formatting rules required
    # for Pydantic parser to work correctly
    partial_variables={
        'format_instruction': parser2.get_format_instructions()
    }
)


# Chain 1:
# feedback → prompt → model → structured output (Feedback object)
classifier_chain = prompt1 | model | parser2


# -------------------- RESPONSE PROMPTS --------------------

# Response if sentiment is positive
prompt2 = PromptTemplate(
    template='Write an appropriate response to this positive feedback:\n{feedback}',
    input_variables=['feedback']
)

# Response if sentiment is negative
prompt3 = PromptTemplate(
    template='Write an appropriate response to this negative feedback:\n{feedback}',
    input_variables=['feedback']
)


# -------------------- BRANCHING LOGIC --------------------

# RunnableBranch works like IF-ELSE inside LangChain

branch_chain = RunnableBranch(

    # Condition 1:
    # If classifier output sentiment == positive
    (
        lambda x: x.sentiment == 'positive',
        prompt2 | model | parser
    ),

    # Condition 2:
    # If sentiment == negative
    (
        lambda x: x.sentiment == 'negative',
        prompt3 | model | parser
    ),

    # Default fallback branch
    RunnableLambda(lambda x: "could not find sentiment")
)


# -------------------- FINAL PIPELINE --------------------

# Full Flow:
#
# feedback
#    ↓
# classifier_chain  → produces Feedback object
#    ↓
# branch_chain      → routes to correct response

chain = classifier_chain | branch_chain


# -------------------- EXECUTION --------------------

print(chain.invoke({
    'feedback': 'This is a beautiful phone'
}))


# -------------------- VISUALIZE EXECUTION GRAPH --------------------

chain.get_graph().print_ascii()
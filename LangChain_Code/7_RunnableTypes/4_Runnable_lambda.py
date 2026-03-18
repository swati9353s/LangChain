# -------------------- Imports --------------------

from langchain_openai import ChatOpenAI          # OpenAI chat model wrapper
from langchain_core.prompts import PromptTemplate # Used to create dynamic prompts
from langchain_core.output_parsers import StrOutputParser # Converts LLM output → plain string
from dotenv import load_dotenv                   # Loads environment variables (.env)
from langchain_core.runnables import (
    RunnableSequence,
    RunnableLambda,
    RunnablePassthrough,
    RunnableParallel
)
import os
# Load API key from .env file (OPENAI_API_KEY)
load_dotenv()


# -------------------- Custom Function --------------------
# This function counts number of words in a given text.
# RunnableLambda allows us to use normal Python functions
# inside LangChain pipelines.

def word_count(text):
    return len(text.split())


# -------------------- Prompt Template --------------------
# Creates a prompt where {topic} will be replaced dynamically.

prompt = PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)


# -------------------- LLM Model --------------------
# Initializes OpenAI chat model
model = ChatOpenAI(
    model='mistralai/Mistral-Small-3.1-24B-Instruct-2503',
    api_key=os.getenv("LLM_API_KEY"),     # API key from .env
    base_url=os.getenv("LLM_BASE_URL")    # Custom provider endpoint
)


# -------------------- Output Parser --------------------
# Converts model output object into a plain string
parser = StrOutputParser()


# -------------------- Joke Generation Chain --------------------
# Flow:
# Prompt → Model → Parser
#
# Input  : {'topic': 'AI'}
# Output : "Generated joke text"

joke_gen_chain = RunnableSequence(
    prompt,
    model,
    parser
)


# -------------------- Parallel Chain --------------------
# After joke is generated, we process it in parallel:
#
# 1. Keep the joke unchanged
# 2. Count number of words in the joke

parallel_chain = RunnableParallel({

    # Passes input forward without modification
    'joke': RunnablePassthrough(),

    # Runs custom Python function on same input
    'word_count': RunnableLambda(word_count)
})


# -------------------- Final Chain --------------------
# Step 1: Generate joke
# Step 2: Send joke to parallel processing

final_chain = RunnableSequence(
    joke_gen_chain,
    parallel_chain
)


# -------------------- Execute Chain --------------------
# Provide topic as input
result = final_chain.invoke({'topic': 'AI'})


# -------------------- Format Final Output --------------------
# result is a dictionary:
# {
#   'joke': '....',
#   'word_count': 15
# }

final_result = "{} \n word count - {}".format(
    result['joke'],
    result['word_count']
)

print(final_result)
# ------------------- Imports -------------------

from langchain_openai import ChatOpenAI              # OpenAI chat model wrapper
from langchain_core.prompts import PromptTemplate     # Used to create dynamic prompts
from langchain_core.output_parsers import StrOutputParser  # Converts LLM output → plain string
from dotenv import load_dotenv                        # Loads API keys from .env file
from langchain_core.runnables import (
    RunnableSequence,
    RunnableParallel,
    RunnablePassthrough
)
import os

# Load environment variables (OPENAI_API_KEY)
load_dotenv()

# ------------------- Prompt 1 -------------------
# This prompt generates a joke based on a topic

prompt1 = PromptTemplate(
    template='Write a joke about {topic}',   # {topic} will be replaced at runtime
    input_variables=['topic']
)

# ------------------- LLM Model -------------------
# Initializes OpenAI chat model
model = ChatOpenAI(
    model='mistralai/Mistral-Small-3.1-24B-Instruct-2503',
    api_key=os.getenv("LLM_API_KEY"),     # API key from .env
    base_url=os.getenv("LLM_BASE_URL")    # Custom provider endpoint
)

# ------------------- Output Parser -------------------
# Converts model response object into plain string text
parser = StrOutputParser()

# ------------------- Prompt 2 -------------------
# This prompt explains a joke generated earlier
prompt2 = PromptTemplate(
    template='Explain the following joke - {text}',
    input_variables=['text']
)

# ------------------- Joke Generation Chain -------------------
# Flow:
# Prompt → LLM → String Output
# Input:  {'topic': 'cricket'}
# Output: "generated joke"
joke_gen_chain = RunnableSequence(
    prompt1,
    model,
    parser
)

# ------------------- Parallel Chain -------------------
# Runs TWO operations simultaneously using SAME input
parallel_chain = RunnableParallel({

    # RunnablePassthrough simply forwards input as-is
    # Here it keeps the joke unchanged
    'joke': RunnablePassthrough(),

    # Second branch explains the joke
    # Input joke text becomes {text} in prompt2
    'explanation': RunnableSequence(
        prompt2,
        model,
        parser
    )
})

# ------------------- Final Chain -------------------
# Step 1: Generate joke
# Step 2: Send joke into parallel chain
#         → one branch keeps joke
#         → second branch explains joke

final_chain = RunnableSequence(
    joke_gen_chain,
    parallel_chain
)

# ------------------- Execute Chain -------------------
# Input topic is given only once
result = final_chain.invoke({'topic': 'cricket'})

# Output will be a dictionary like:
# {
#   "joke": "....generated joke....",
#   "explanation": "....explanation of joke...."
# }
print(result)
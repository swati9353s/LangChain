# Import required LangChain components
from langchain_openai import ChatOpenAI          # OpenAI chat model wrapper
from langchain_core.prompts import PromptTemplate # Used to create reusable prompt templates
from langchain_core.output_parsers import StrOutputParser  # Converts model output into plain string
from dotenv import load_dotenv                   # Loads environment variables (.env file)
from langchain_core.runnables import RunnableSequence, RunnableParallel
import os

# Load environment variables (like OPENAI_API_KEY)
load_dotenv()

# ------------------- Prompt Templates -------------------

# Prompt to generate a Twitter-style post
prompt1 = PromptTemplate(
    template='Generate a tweet about {topic}',   # Placeholder {topic} will be replaced at runtime
    input_variables=['topic']                    # Required input variable
)

# Prompt to generate a LinkedIn-style post
prompt2 = PromptTemplate(
    template='Generate a Linkedin post about {topic}',
    input_variables=['topic']
)

# ------------------- LLM Model -------------------

# Initialize OpenAI chat model
# Uses API key from environment variables
model = ChatOpenAI(
    model='mistralai/Mistral-Small-3.1-24B-Instruct-2503',
    api_key=os.getenv("LLM_API_KEY"),     # API key from .env
    base_url=os.getenv("LLM_BASE_URL")    # Custom provider endpoint
)
# ------------------- Output Parser -------------------

# Converts LLM response object into a simple string
parser = StrOutputParser()

# ------------------- Parallel Chain -------------------

# RunnableParallel runs multiple chains at the SAME time in the dictionary format
# Each key represents one independent workflow
parallel_chain = RunnableParallel({

    # First workflow:
    # prompt1 → model → parser
    # Generates a tweet
    'tweet': RunnableSequence(prompt1, model, parser),

    # Second workflow:
    # prompt2 → model → parser
    # Generates a LinkedIn post
    'linkedin': RunnableSequence(prompt2, model, parser)
})

# ------------------- Execute Parallel Chain -------------------

# Input is passed to BOTH chains simultaneously
result = parallel_chain.invoke({'topic': 'AI'})

# result is a dictionary:
# {
#   "tweet": "...generated tweet...",
#   "linkedin": "...generated linkedin post..."
# }

# Print outputs separately
print(result['tweet'])
print(result['linkedin'])
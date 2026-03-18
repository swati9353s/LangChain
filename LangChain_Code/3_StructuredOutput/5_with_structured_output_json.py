# ---------------------------------------------------
# Import Required Libraries
# ---------------------------------------------------

# ChatOpenAI -> Used to connect LangChain with OpenAI-compatible APIs
from langchain_openai import ChatOpenAI

# load_dotenv -> Loads API keys from .env file
from dotenv import load_dotenv

# Streamlit imported (useful if you later build UI)
import streamlit as st

import os

# Typing utilities (not strictly required here but useful for schemas)
from typing import TypedDict, Annotated, Optional, Literal

# Pydantic (not used directly here but commonly used for schema creation)
from pydantic import BaseModel, Field


# ---------------------------------------------------
# STEP 1: Load Environment Variables
# ---------------------------------------------------
# Reads API key and base URL from .env file
load_dotenv()


# ---------------------------------------------------
# STEP 2: Create the LLM Model
# ---------------------------------------------------
# We are using an OpenAI-compatible endpoint.
# This can be OpenRouter, TogetherAI, or any hosted model API.

model = ChatOpenAI(
    model='mistralai/Mistral-Small-3.1-24B-Instruct-2503',
    api_key=os.getenv("LLM_API_KEY"),     # API key from .env
    base_url=os.getenv("LLM_BASE_URL")    # Custom provider endpoint
)


# ---------------------------------------------------
# STEP 3: Define Structured Output Schema (JSON Schema)
# ---------------------------------------------------
# Instead of Pydantic class, we manually define JSON schema.
# This tells the LLM EXACTLY how output should be formatted.

json_schema = {
  "title": "Review",
  "type": "object",
  "properties": {

    # List of important themes discussed in review
    "key_themes": {
      "type": "array",
      "items": {"type": "string"},
      "description": "Write down all the key themes discussed in the review in a list"
    },

    # Short summary of review
    "summary": {
      "type": "string",
      "description": "A brief summary of the review"
    },

    # Sentiment restricted to fixed values
    "sentiment": {
      "type": "string",
      "enum": ["pos", "neg"],
      "description": "Return sentiment of the review either negative or positive"
    },

    # Optional pros list
    "pros": {
      "type": ["array", "null"],
      "items": {"type": "string"},
      "description": "Write down all the pros inside a list"
    },

    # Optional cons list
    "cons": {
      "type": ["array", "null"],
      "items": {"type": "string"},
      "description": "Write down all the cons inside a list"
    },

    # Optional reviewer name
    "name": {
      "type": ["string", "null"],
      "description": "Write the name of the reviewer"
    }
  },

  # These fields MUST appear in output
  "required": ["key_themes", "summary", "sentiment"]
}


# ---------------------------------------------------
# STEP 4: Attach Schema to Model
# ---------------------------------------------------
# with_structured_output() forces the model
# to generate VALID JSON matching the schema.

structured_model = model.with_structured_output(json_schema)


# ---------------------------------------------------
# STEP 5: Send Review Text to Model
# ---------------------------------------------------
# The model extracts structured information
# from the unstructured product review.

result = structured_model.invoke("""
I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it’s an absolute powerhouse! 
The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I’m gaming, multitasking, or editing photos. 
The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. 
What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. 
Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.

However, the weight and size make it a bit uncomfortable for one-handed use. 
Also, Samsung’s One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? 
The $1,300 price tag is also a hard pill to swallow.

Pros:
Insanely powerful processor (great for gaming and productivity)
Stunning 200MP camera with incredible zoom capabilities
Long battery life with fast charging
S-Pen support is unique and useful

Review by Nitish Singh
""")


# ---------------------------------------------------
# STEP 6: Print Structured Output
# ---------------------------------------------------
# Output will be validated JSON converted into Python object/dict
print(result)

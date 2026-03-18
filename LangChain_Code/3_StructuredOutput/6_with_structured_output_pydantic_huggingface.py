# Load environment variables from .env file
# (Used to store API keys securely instead of hardcoding them)
from dotenv import load_dotenv

# Typing utilities
# Optional  -> field may or may not be present
# Literal   -> restricts values to fixed options
from typing import Optional, Literal

# Pydantic is used to define a structured schema (data validation)
from pydantic import BaseModel, Field

# LangChain HuggingFace integration
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# Load environment variables (HF token, etc.)
load_dotenv()


# ---------------------------------------------------
# STEP 1: Create the LLM (Language Model)
# ---------------------------------------------------
# NOTE:
# Very small/open-source models sometimes DO NOT properly
# support structured outputs (JSON schema following).
# They may generate invalid JSON or throw errors.

llm = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # HuggingFace model
    task="text-generation"                         # Task type
)

# Wrap HuggingFace model inside LangChain chat interface
model = ChatHuggingFace(llm=llm)


# ---------------------------------------------------
# STEP 2: Define Structured Output Schema using Pydantic
# ---------------------------------------------------
# This schema tells the LLM EXACTLY how the output should look.

class Review(BaseModel):

    # List of important topics discussed in review
    key_themes: list[str] = Field(
        description="Write down all the key themes discussed in the review in a list"
    )

    # Short summary of the review
    summary: str = Field(
        description="A brief summary of the review"
    )

    # Sentiment restricted to ONLY these values
    sentiment: Literal["pos", "neg"] = Field(
        description="Return sentiment of the review either negative or positive"
    )

    # Optional fields (may or may not appear in output)
    pros: Optional[list[str]] = Field(
        default=None,
        description="Write down all the pros inside a list"
    )

    cons: Optional[list[str]] = Field(
        default=None,
        description="Write down all the cons inside a list"
    )

    name: Optional[str] = Field(
        default=None,
        description="Write the name of the reviewer"
    )


# ---------------------------------------------------
# STEP 3: Attach Schema to Model
# ---------------------------------------------------
# with_structured_output() forces the model
# to return output matching the Review schema.

structured_model = model.with_structured_output(Review)


# ---------------------------------------------------
# STEP 4: Invoke the model
# ---------------------------------------------------
# We instruct the model to return ONLY valid JSON.
# (Important when using structured outputs)

result = structured_model.invoke("""
Extract information and RETURN ONLY VALID JSON.
Do not add explanation.
""")


# ---------------------------------------------------
# STEP 5: Print Structured Result
# ---------------------------------------------------
# Output will be a validated Pydantic object
print(result)

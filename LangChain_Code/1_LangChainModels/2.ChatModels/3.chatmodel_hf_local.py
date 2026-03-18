# We can use Hugging Face models in two ways:
# 1) Through cloud API (Hugging Face Inference Endpoint)
# 2) Through local deployment (running the model on our own machine)
#
# This file demonstrates LOCAL deployment of a Hugging Face model

from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from dotenv import load_dotenv
import os

# Load environment variables (not strictly required for local models,
# but good practice if you use API keys elsewhere)
load_dotenv()

# Create a local Hugging Face pipeline using a small open-source model
llm = HuggingFacePipeline.from_model_id(
    model_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',  # Lightweight model suitable for local use
    task='text-generation',                        # Task: generate text responses
    pipeline_kwargs=dict(
        temperature=0.7,     # Controls randomness (higher = more creative)
        max_new_tokens=100,  # Maximum number of tokens to generate in response
    )
)

# Wrap the local pipeline into LangChain's chat interface
model = ChatHuggingFace(llm=llm)

# Send a question to the locally running model
result = model.invoke("What is the capital of France?")

# Print the model's reply
print(result.content)

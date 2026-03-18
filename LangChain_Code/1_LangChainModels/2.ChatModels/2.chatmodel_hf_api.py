# Import ChatHuggingFace -> this wraps a Hugging Face model into a chat-style interface (like ChatGPT)
from langchain_huggingface import ChatHuggingFace   

# Import HuggingFaceEndpoint -> used to connect to Hugging Face’s cloud API (Inference endpoint)
from langchain_huggingface import HuggingFaceEndpoint  

# Import load_dotenv -> helps load environment variables from the .env file (like API keys)
from dotenv import load_dotenv           

import os   # Used to read environment variables in Python

# Load all variables from the .env file into the program
load_dotenv(r'C:\Users\smi68\Desktop\My_Learning\Artificial-Intelligence\LangChain\LangChain_Code\.env')

# Create a connection to a Hugging Face hosted model (cloud-based)
llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1-0528",   # The Hugging Face model we want to use
    task="text-generation",                  # Tells HF that we want text generation
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_KEY"),  # API key for authentication
)

# Convert the Hugging Face model into a chat-style model for LangChain
model = ChatHuggingFace(llm=llm)

# Send a prompt (question) to the model
result = model.invoke("What is the capital of France?")

# Print only the text response from the model
print(result.content)

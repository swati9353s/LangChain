from langchain_openai import ChatOpenAI   # OpenAI-compatible chat model interface from LangChain
from dotenv import load_dotenv           # Used to load environment variables from .env file
import os                               

# Load all variables written in the .env file into the environment
# Example: LLM_API_KEY, LLM_BASE_URL
load_dotenv()

# -------------------------------------------------------
# NOTE:
# We are NOT calling OpenAI's official servers.
# We are using our OWN LLM server (OpenAI-compatible API)
# -------------------------------------------------------

# Create a chat model instance using ChatOpenAI interface
# Even though the class name is ChatOpenAI, it can work with
# any model that follows OpenAI API format (like Mistral here)

model = ChatOpenAI(
    model='mistralai/Mistral-Small-3.1-24B-Instruct-2503',  # Model name running on your LLM server
    api_key=os.getenv("LLM_API_KEY"),                       # API key taken from .env file
    base_url=os.getenv("LLM_BASE_URL"),                      # Your custom LLM server URL
    #When the temperature is set near to 0 every time we get the same answer
    #when it is set near to 1 we get different answers
    temperature=1.8,
    #we can set the maximum token limit for the response
    max_completion_tokens=10
)

# Send a prompt (user message) to the model
# invoke() runs the request and gets the AI response
result = model.invoke("What is the capital of France?")

# Print the AI response
# print(result)

# To Print the exact output we use result.content
print(result.content)

#The benefit of using the langchain is the same code can used for the antropic models, google apis etc

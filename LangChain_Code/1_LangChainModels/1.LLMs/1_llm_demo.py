from langchain_openai import OpenAI
from dotenv import load_dotenv

#we will load the env variable from the env file
load_dotenv()

#call the model
llm = OpenAI(model='gpt-3.5-turbo-instruct')

#invoke the model
#invoke methods takes string as an input
#It is the normal llm so it takes string as an input and gives string as an output
result = llm.invoke("What is the capital of France?")

print(result)
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
import os

load_dotenv()  
# Loads environment variables so API keys and endpoints
# can be accessed securely using os.getenv().

model1 = ChatOpenAI(
    model='mistralai/Mistral-Small-3.1-24B-Instruct-2503',
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL")
)

model2 = ChatOpenAI(
    model='mistralai/Mistral-Small-3.1-24B-Instruct-2503',
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL")
)

# ---------------- PROMPT 1 ----------------
# Generates simplified notes from the given text.
# {text} will be replaced at runtime.
prompt1 = PromptTemplate(
    template='Generate short and simple notes from the following text \n {text}',
    input_variables=['text']
)

# ---------------- PROMPT 2 ----------------
# Generates quiz-style question answers from the same input text.
prompt2 = PromptTemplate(
    template='Generate 5 short question answers from the following text \n {text}',
    input_variables=['text']
)

# ---------------- PROMPT 3 ----------------
# Combines outputs coming from two parallel executions.
# {notes} and {quiz} will automatically receive outputs
# produced by the parallel chain.
prompt3 = PromptTemplate(
    template='Merge the provided notes and quiz into a single document \n notes -> {notes} and quiz -> {quiz}',
    input_variables=['notes', 'quiz']
)

# ---------------- OUTPUT PARSER ----------------
# StrOutputParser extracts only the generated text
# from model response objects.
# This ensures clean strings are passed between steps.
parser = StrOutputParser()

# ---------------- PARALLEL EXECUTION ----------------
# RunnableParallel runs multiple chains at the SAME time.
#
# Both branches receive the SAME input dictionary.
# Output becomes:
# {
#   "notes": <generated notes>,
#   "quiz": <generated quiz>
# }
#
# Benefit:
# - Faster execution (parallel processing)
# - Independent tasks can run simultaneously
# - Clean structured outputs mapped by keys
parallel_chain = RunnableParallel({
    'notes': prompt1 | model1 | parser,
    'quiz': prompt2 | model2 | parser
})

# ---------------- MERGE CHAIN ----------------
# Takes outputs from parallel_chain and injects them
# into prompt3 automatically.
# Produces a final merged document.
merge_chain = prompt3 | model1 | parser

# ---------------- FINAL PIPELINE ----------------
# First executes parallel tasks,
# then sends combined result to merge step.
chain = parallel_chain | merge_chain


# ---------------- INPUT TEXT ----------------
# Shared input passed to both parallel branches.
text = """
Support vector machines (SVMs) are a set of supervised learning methods used for classification, regression and outliers detection.

The advantages of support vector machines are:

Effective in high dimensional spaces.

Still effective in cases where number of dimensions is greater than the number of samples.

Uses a subset of training points in the decision function (called support vectors), so it is also memory efficient.

Versatile: different Kernel functions can be specified for the decision function. Common kernels are provided, but it is also possible to specify custom kernels.

The disadvantages of support vector machines include:

If the number of features is much greater than the number of samples, avoid over-fitting in choosing Kernel functions and regularization term is crucial.

SVMs do not directly provide probability estimates, these are calculated using an expensive five-fold cross-validation (see Scores and probabilities, below).

The support vector machines in scikit-learn support both dense (numpy.ndarray and convertible to that by numpy.asarray) and sparse (any scipy.sparse) sample vectors as input. However, to use an SVM to make predictions for sparse data, it must have been fit on such data. For optimal performance, use C-ordered numpy.ndarray (dense) or scipy.sparse.csr_matrix (sparse) with dtype=float64.
"""

# invoke():
# Executes the entire workflow:
# 1. Same text sent to both parallel branches.
# 2. Notes and quiz generated simultaneously.
# 3. Outputs merged into final document.
result = chain.invoke({'text': text})

# Final merged output.
print(result)

# ---------------- GRAPH VISUALIZATION ----------------
# get_graph():
# Builds execution graph of the runnable pipeline.
#
# print_ascii():
# Displays a text diagram showing how data flows
# through parallel and sequential components.
chain.get_graph().print_ascii()
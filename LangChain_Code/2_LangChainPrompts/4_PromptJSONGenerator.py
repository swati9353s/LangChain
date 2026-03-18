# Import PromptTemplate class
# Used to create reusable and dynamic prompts in LangChain
from langchain_core.prompts import PromptTemplate


# ---------------------------------------------------------
# Create a PromptTemplate
# ---------------------------------------------------------
template = PromptTemplate(

    # Template string containing placeholders
    # These placeholders will be replaced dynamically at runtime
    template="""Explain the research paper "{paper}" 
    in a {style} style and make it {length}""",

    # List of variables expected inside the template
    # LangChain will validate that all variables are provided
    input_variables=["paper", "style", "length"],

    # Enables template validation
    # If placeholders and input_variables do not match,
    # LangChain throws an error immediately
    validate_template=True,
)


# ---------------------------------------------------------
# Save the prompt template into a JSON file
# This allows separation of:
#   1. Prompt engineering
#   2. Application logic
#
# The saved file can later be loaded using load_prompt()
# ---------------------------------------------------------
template.save("promptTemplate.json")

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from typing import TypedDict, Annotated, Optional, Literal

# Load environment variables (.env file)
load_dotenv()

# ---------------------------------------------------
# STEP 1: Initialize LLM Model
# ---------------------------------------------------
# ChatOpenAI wrapper allows us to use OpenAI-compatible APIs
# Here we are using a hosted Mistral model.

model = ChatOpenAI(
    model='mistralai/Mistral-Small-3.1-24B-Instruct-2503',
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL")
)

# ---------------------------------------------------
# STEP 2: Define Structured Output Schema using TypedDict
# ---------------------------------------------------
# total=False → all fields become OPTIONAL.
# LLM may or may not return every field.

class Review(TypedDict, total=False):

    # Annotated adds description (instruction) for the LLM.
    # It acts like inline documentation for the model.

    key_themes: Annotated[
        list[str],
        "Write down all the key themes discussed in the review in a list"
    ]

    summary: Annotated[
        str,
        "A brief summary of the review"
    ]

    # Literal restricts allowed output values.
    sentiment: Annotated[
        Literal["pos", "neg"],
        "Return sentiment as either pos or neg"
    ]

    # Optional fields → may be missing in LLM output
    pros: Annotated[
        Optional[list[str]],
        "Write down all the pros inside a list"
    ]

    cons: Annotated[
        Optional[list[str]],
        "Write down all the cons inside a list"
    ]

    name: Annotated[
        Optional[str],
        "Write the name of the reviewer"
    ]


# ---------------------------------------------------
# STEP 3: Enable Structured Output
# ---------------------------------------------------
# LangChain automatically converts this schema into
# instructions for the LLM and forces JSON output.

structured_model = model.with_structured_output(Review)


# ---------------------------------------------------
# STEP 4: Invoke Model
# ---------------------------------------------------
result = structured_model.invoke("""
I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it’s an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I’m gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.

However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung’s One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.

Pros:
Insanely powerful processor
Stunning 200MP camera
Long battery life
S-Pen support

Review by Nitish Singh
""")


# ---------------------------------------------------
# STEP 5: Access Structured Data
# ---------------------------------------------------
# TypedDict returns a normal Python dictionary.
print(result['pros'])

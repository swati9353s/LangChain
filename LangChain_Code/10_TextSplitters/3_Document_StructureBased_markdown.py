# -------------------- Imports --------------------

# RecursiveCharacterTextSplitter splits text intelligently
# Language enum allows language-aware splitting (Markdown, Python, HTML, etc.)
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language


# -------------------- Sample Markdown Text --------------------

# Markdown formatted project description
# Contains headings, lists, and code blocks
text = """
# Project Name: Smart Student Tracker

A simple Python-based project to manage and track student data, including their grades, age, and academic status.

## Features

- Add new students with relevant info
- View student details
- Check if a student is passing
- Easily extendable class-based design

## 🛠 Tech Stack

- Python 3.10+
- No external dependencies

## Getting Started

1. Clone the repo  
   ```bash
   git clone https://github.com/your-username/student-tracker.git
"""
    

# -------------------- Initialize Language-Aware Splitter --------------------

# from_language() creates a splitter optimized for a specific format.
# Here we use MARKDOWN so it tries to split at:
#   headings (#, ##)
#   lists
#   code blocks
# instead of randomly cutting text.
splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.MARKDOWN,

    # Maximum characters allowed per chunk
    chunk_size=200,

    # No overlap between chunks
    chunk_overlap=0,
)


# -------------------- Split the Text --------------------

# split_text() divides markdown into meaningful sections
chunks = splitter.split_text(text)


# -------------------- Output --------------------

# Number of chunks created
print(len(chunks))

# Print first chunk
print(chunks[0])
# -------------------- Imports --------------------

# RecursiveCharacterTextSplitter:
# Splits text intelligently using structure first, then characters.
# Language enum allows language-aware splitting rules.
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language


# -------------------- Python Code Text --------------------

# A Python program stored as plain text.
# We want to split it without breaking functions/classes randomly.
text = """
class Student:
    def __init__(self, name, age, grade):
        self.name = name
        self.age = age
        self.grade = grade  # Grade is a float (like 8.5 or 9.2)

    def get_details(self):
        return self.name"

    def is_passing(self):
        return self.grade >= 6.0


# Example usage
student1 = Student("Aarav", 20, 8.2)
print(student1.get_details())

if student1.is_passing():
    print("The student is passing.")
else:
    print("The student is not passing.")
"""


# -------------------- Initialize Python-aware Splitter --------------------

# from_language(Language.PYTHON) makes the splitter
# understand Python structure such as:
#   - class definitions
#   - function definitions
#   - logical blocks
splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    
    # Maximum characters allowed per chunk
    chunk_size=300,
    
    # No overlap between chunks
    chunk_overlap=0,
)


# -------------------- Perform Splitting --------------------

# split_text() divides the code into meaningful chunks
chunks = splitter.split_text(text)


# -------------------- Output --------------------

# Number of chunks created
print(len(chunks))

# Print second chunk
print(chunks[1])
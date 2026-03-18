from typing import TypedDict

# ---------------------------------------------------
# STEP 1: Define a TypedDict
# ---------------------------------------------------
# person is inheriting from TypedDict.
# This defines the EXPECTED structure of a dictionary.
# It does NOT create a normal class object.
# It only provides type hints (structure checking).

class person(TypedDict):
    name: str   # 'name' key must be a string
    age: int    # 'age' key must be an integer


# ---------------------------------------------------
# STEP 2: Create a dictionary following the schema
# ---------------------------------------------------
# Here we are declaring that new_person must follow
# the structure defined in 'person'.

new_person: person = {
    'name': 'Swati',
    'age': 22
}

# ---------------------------------------------------
# STEP 3: Print the dictionary
# ---------------------------------------------------
print(new_person)

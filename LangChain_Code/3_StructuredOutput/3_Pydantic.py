from pydantic import BaseModel, EmailStr, Field
from typing import Optional


# ---------------------------------------------------
# STEP 1: Create Pydantic Model
# ---------------------------------------------------
# BaseModel allows us to define structure + validation rules

class student(BaseModel):

    # Default value is assigned.
    # If name is not provided, "swati" will be used automatically.
    name: str = 'swati'

    # Optional field → value may or may not be provided.
    # Default value is None.
    age: Optional[int] = None

    # EmailStr performs automatic email validation.
    # This field is REQUIRED because no default value is given.
    email: EmailStr

    # Field() allows adding constraints and metadata.
    # gt=0  → value must be greater than 0
    # lt=10 → value must be less than 10
    # default value is 5 if not provided
    cgpa: float = Field(
        gt=0,
        lt=10,
        default=5,
        description="A decimal value representing the cgpa of a person"
    )


# ---------------------------------------------------
# STEP 2: Create Input Dictionary
# ---------------------------------------------------
# age is given as string → Pydantic performs implicit type conversion
# and converts it into integer automatically.

new_student = {
    'name': 'swati',
    'age': '22',          # string → automatically converted to int
    'email': 'abc@gmail.com',
    'cgpa': 8
}


# ---------------------------------------------------
# STEP 3: Dictionary Unpacking (**)
# ---------------------------------------------------
# **new_student unpacks dictionary into keyword arguments.
#
# Equivalent to:
# student(name='swati', age='22', email='abc@gmail.com', cgpa=8)

Student = student(**new_student)


# NOTE:
# email is mandatory because EmailStr has no default value.
# If we create an empty dictionary and try:
#
# new_empty_student = {}
# Student1 = student(**new_empty_student)
#
# → Pydantic will raise ValidationError (email missing).


# ---------------------------------------------------
# STEP 4: Print Pydantic Object
# ---------------------------------------------------
print(Student)

# Output is a Pydantic object (NOT a dictionary)
print(type(Student))


# ---------------------------------------------------
# STEP 5: Convert Pydantic Object → Dictionary
# ---------------------------------------------------
# Useful when sending data to APIs
student_dict = dict(Student)
print(student_dict['name'])


# ---------------------------------------------------
# STEP 6: Convert Pydantic Object → JSON
# ---------------------------------------------------
# Useful when storing data in databases or transmitting over network
student_json = Student.model_dump_json()
print(student_json)

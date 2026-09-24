students = [
    {"name": "Hermione", "house": "Gryffindor", "patronous": "otter"},
    {"name": "Harry", "house": "Gryffindor", "patronous":"Stag"},
    {"name": "Ron", "house": "Gryffindor", "patronous":"Jack Russell terrir"},
    {"name": "Graco", "house": "Slytherin", "patronous": None},
]
for student in students:
    print(student["name"], student["house"], student["patronous"], sep = ", ")



import json
import random
from datetime import date, timedelta

import unify

client = unify.Unify("o1@openai", cache=True)
response = client.generate(
"""
Please generate exactly 100 imaginary student first names, last names, emails, 
and their gender (male or female), and present this in .csv format, in alphabetical 
order from first names beginning with A down to first names beginning with Z. The 
final part of your response should be formatted as follows, starting on a new line:
first_name,last_name,email,gender
{first_name_1},{last_name_1},{email_1},{gender_1}
...
{first_name_100},{last_name_100},{email_100},{gender_100}
"""
)
students = response.split("first_name,last_name,email,gender")[-1]
students = [
    dict(zip(("first_name", "last_name", "email", "gender"), row.split(",")))
    for row in students.split("\n") if row != ""
]
num_students = len(students)


def random_date(start_date: date, end_date: date) -> str:
    delta = end_date - start_date
    days_between = delta.days
    random_days = random.randint(0, days_between)
    return (start_date + timedelta(days=random_days)).isoformat()


for student in students:
    student["date_of_birth"] = random_date(date(1971, 1, 1), date(2010, 1, 1))

with open('students.json', 'w+') as f:
    f.write(json.dumps(students, indent=4))

import requests
import json

url = 'https://nd0821-c3-machado.onrender.com/'
census = {
    "age": 30,
    "workclass": "State-gov",
    "fnlgt": 141297,
    "education": "Bachelors",
    "education_num": 13,
    "marital_status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Husband",
    "race": "Asian-Pac-Islander",
    "sex": "Male",
    "capital_gain": 0,
    "capital_loss": 0,
    "hours_per_week": 40,
    "native_country": "India"
    }

response = requests.post(url, data=json.dumps(census))

print('status code:', response.status_code)
print('salary:', response.json())

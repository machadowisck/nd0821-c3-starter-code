import requests
import json

url = 'https://nd0821-c3-machado.onrender.com/'
census = {
    'age': 39,
    'workclass': 'State-gov',
    'fnlgt': 77516,
    'education': 'Bachelors',
    'education-num': 13,
    'marital-status': 'Never-married',
    'occupation': 'Adm-clerical',
    'relationship': 'Not-in-family',
    'race': 'White',
    'sex': 'Male',
    'capital-gain': 2174,
    'capital-loss': 0,
    'hours-per-week': 40,
    'native-country': 'United-States'
    }

response = requests.post(url, data=json.dumps(census))

print('status code:', response.status_code)
print('salary:', response.json())

import requests
import json

url = 'http://127.0.0.1:8000'
census = {"age": 30,
          "workclass": "State-gov",
          "fnlgt": 141297,
          "education": "Bachelors",
          "education-num": 13,
          "marital-status": "Married-civ-spouse",
          "occupation": "Prof-specialty",
          "relationship": "Husband",
          "race": "Asian-Pac-Islander",
          "sex": "Male",
          "capital-gain": 0,
          "capital-loss": 0,
          "hours-per-week": 40,
          "native-country": "India"}


print('#############################################')
print('/ GET')
response = requests.get(url)
print('status code:', response.status_code)
# print('salary:', response.json())
print('Response Headers:', response.headers)
print('text:', response.text)

print('#############################################')
print('/ POST')
headers = {'Content-Type': 'application/json'}
#  response = requests.post(url, data=json.dumps(census), headers=headers)
response = requests.post(url, json=census)

print('status code:', response.status_code)
# print('salary:', response.json())
print("Request Headers:", response.request.headers)
print("Request Body:", response.request.body)
print('Response Headers:', response.headers)
print('text:', response.text)
print('more: ', response.reason)

"""
def test_post2():
    data = [{
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
    }]
    r = TestClient(app).post("/", json=data)
    assert r.status_code == 200
    assert r.json() == ">50K"
    """

import json
from fastapi.testclient import TestClient
from starter.main import app

# import sys
# sys.path.insert(0, 'starter')


def test_get_root():
    r = TestClient(app).get("/")
    assert r.status_code == 200
    assert r.json() == "Welcome to the Census income class predictor API!"


def test_post():
    data = {
        "age": 40,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education_num": 13,
        "marital_status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Female",
        "capital_gain": 2174,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States"
    }
    r = TestClient(app).post("/", data=json.dumps(data))
    assert r.status_code == 200
    assert r.json() == "<=50K"


def test_post2():
    data = {
        'age': 32,
        'workclass': 'Private',
        'fnlgt': 114937,
        'education': 'Assoc-acdm',
        'education-num': 13,
        'marital-status': 'Married-civ-spouse',
        'occupation': 'Adm-clerical',
        'relationship': 'Husband',
        'race': 'White',
        'sex': 'Male',
        'capital-gain': 0,
        'capital-loss': 0,
        'hours-per-week': 40,
        'native-country': 'United-States'
    }
    r = TestClient(app).post("/", data=json.dumps(data))
    assert r.status_code == 200
    assert r.json() == ">50K"

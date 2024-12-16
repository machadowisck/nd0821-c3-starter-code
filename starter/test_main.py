# import json
from fastapi.testclient import TestClient
from .main import app

# import sys
# sys.path.insert(0, 'starter')


def test_get_root():
    r = TestClient(app).get("/")
    assert r.status_code == 200
    assert r.json() == "Welcome to the Census income class predictor API!"


def test_post():
    """
    data = [{
        "age": 39,
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
    }]
    """
    data = {
        "age": 39,
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
    r = TestClient(app).post("/", json=data)
    assert r.status_code == 200
    assert r.json() == "<=50K"


def test_post2():
    '''
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
    '''
    data = {
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
    r = TestClient(app).post("/", json=data)
    assert r.status_code == 200
    assert r.json() == ">50K"

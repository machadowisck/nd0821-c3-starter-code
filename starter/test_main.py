# import json
from fastapi.testclient import TestClient
from .main import app

# import sys
# sys.path.insert(0, 'starter')


def test_get_root():
    r = TestClient(app).get("/")
    assert r.status_code == 200
    assert r.json() == "Welcome to the Census income class predictor API!"


def test_post_le():
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
        "sex": "Male",
        "capital_gain": 2174,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States"
        }]
    r = TestClient(app).post("/", json=data)
    assert r.status_code == 200
    assert r.json() == "<=50K"


def test_post_gt():
    # 52,Self-emp-not-inc,209642,HS-grad,9,Married-civ-spouse,Exec-managerial,Husband,White,Male,0,0,45,United-States,>50K
    data = [{
        "age": 52,
        "workclass": "Self-emp-not-inc",
        "fnlgt": 209642,
        "education": "HS-grad",
        "education_num": 9,
        "marital_status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 45,
        "native_country": "United-States"
    }]
    r = TestClient(app).post("/", json=data)
    assert r.status_code == 200
    assert r.json() == ">50K"

"""
Tests for API of the income prediction service.
Author: Andrei Goponenko
Date: 5 April 2025
"""

from fastapi.testclient import TestClient

from main import app


client = TestClient(app)


def test_say_hi():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"greetings": "Hello!"}


def test_negative_inference():
    item = {
        'age': 50,
        'workclass': 'Self-emp-not-inc',
        'fnlgt': 83311,
        'education': 'Bachelors',
        'education-num': 13,
        'marital-status': 'Married-civ-spouse',
        'occupation': 'Exec-managerial',
        'relationship': 'Husband',
        'race': 'White',
        'sex': 'Male',
        'capital-gain': 0,
        'capital-loss': 0,
        'hours-per-week': 13,
        'native-country': 'United-States',
        'salary': '<=50K'
    }
    r = client.post("/predict", json=item)
    assert r.status_code == 200
    assert r.json() == {"prediction": "<=50K"}


def test_positive_inference():
    item = {
        'age': 42,
        'workclass': 'Private',
        'fnlgt': 159449,
        'education': 'Bachelors',
        'education-num': 13,
        'marital-status': 'Married-civ-spouse',
        'occupation': 'Exec-managerial',
        'relationship': 'Husband',
        'race': 'White',
        'sex': 'Male',
        'capital-gain': 5178,
        'capital-loss': 0,
        'hours-per-week': 40,
        'native-country': 'United-States',
        'salary': '>50K'
    }
    r = client.post("/predict", json=item)
    assert r.status_code == 200
    assert r.json() == {"prediction": ">50K"}

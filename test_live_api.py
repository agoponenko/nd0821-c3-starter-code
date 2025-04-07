"""
Test for deployed service of the income prediction service.
Author: Andrei Goponenko
Date: 5 April 2025
"""

import requests

app_url = "https://income-predictor-f1bz.onrender.com/"

# Test the GET method
request_1 = requests.get(app_url)
assert request_1.status_code == 200
print(request_1.json())

# Test the POST method
data = {
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
request_2 = requests.post(f"{app_url}predict", json=data)

assert request_2.status_code == 200
print(request_2.json())
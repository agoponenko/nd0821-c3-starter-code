"""
API for the income prediction service.
Author: Andrei Goponenko
Date: 5 April 2025
"""

import os
from pathlib import Path
import sys

from fastapi import FastAPI
import pandas as pd
import pickle
from pydantic import BaseModel, Field

sys.path.insert(0, "starter/starter")

from ml.data import process_data
from ml.model import inference


if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")


# Upload artifacts necessary to run preprocessing and make inference
cur_folder_path = Path(__file__).resolve().parent
model = pickle.load(open(cur_folder_path / "model/model.pickle", "rb"))
encoder = pickle.load(open(cur_folder_path / "model/encoder.pickle", "rb"))
lb = pickle.load(open(cur_folder_path / "model/lb.pickle", "rb"))


class Item(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(alias='education-num')
    marital_status: str = Field(alias='marital-status')
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias='capital-gain')
    capital_loss: int = Field(alias='capital-loss')
    hours_per_week: int = Field(alias='hours-per-week')
    native_country: str = Field(alias='native-country')
    salary: str

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "age": 39,
                    "workclass": "State-gov",
                    "fnlgt": 77516,
                    "education": "Bachelors",
                    "education-num": 13,
                    "marital-status": "Never-married",
                    "occupation": "Adm-clerical",
                    "relationship": "Not-in-family",
                    "race": "White",
                    "sex": "Male",
                    "capital-gain": 2147,
                    "capital-loss": 0,
                    "hours-per-week": 40,
                    "native-country": "United-States",
                    "salary": "<=50K"
                }
            ]
        }


# Initialise App, create GET and POST
app = FastAPI()


@app.get("/")
async def say_hi():
    return {"greetings": "Hello!"}


@app.post("/predict")
async def predict(item: Item):

    cat_features = [
        "workclass",
        "education",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native_country",
    ]

    df = pd.DataFrame(item.dict(), index=[0])

    X_test, _, _, _ = process_data(
        df,
        categorical_features=cat_features,
        label='salary',
        training=False,
        encoder=encoder,
        lb=lb)

    pred = lb.inverse_transform(inference(model, X_test))[0]

    return {"prediction": pred}

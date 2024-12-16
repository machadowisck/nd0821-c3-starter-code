# Put the code for your API here.

import os
import pickle
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field, ConfigDict
from starter.starter.ml.data import process_data
from starter.starter.ml.model import inference


app = FastAPI()
file_dir = os.path.dirname(__file__)
model_path = os.path.join(file_dir, 'model/rf_model.pkl')
encoder_path = os.path.join(file_dir, 'model/encoder.pkl')
lb_path = os.path.join(file_dir, 'model/lb.pkl')
model = pickle.load(open(model_path, 'rb'))
encoder = pickle.load(open(encoder_path, 'rb'))
lb = pickle.load(open(lb_path, 'rb'))


def field_alias(string: str):
    return string.replace('_', '-')


class Census(BaseModel):
    # Using the first row of census.csv as sample
    age: int = Field(None, example=40)
    workclass: str = Field(None, example='State-gov')
    fnlgt: int = Field(None, example=88516)
    education: str = Field(None, example='Bachelors')
    education_num: int = Field(None, alias='education_num', example=13)
    marital_status: str = Field(None,
                                alias='marital_status',
                                example='Never-married')
    occupation: str = Field(None, example='Adm-clerical')
    relationship: str = Field(None, example='Not-in-family')
    race: str = Field(None, example='Black')
    sex: str = Field(None, example='Female')
    capital_gain: int = Field(None, alias='capital_gain', example=2174)
    capital_loss: int = Field(None, alias='capital_loss', example=0)
    hours_per_week: int = Field(None, alias='hours_per_week', example=40)
    native_country: str = Field(None, alias='native_country', example='Peru')
    model_config = ConfigDict(alias_generator=field_alias)  # allow_population_by_field_name=True


@app.get("/")
async def home_page():
    return "Welcome to the Census income class predictor API!"


@app.post("/")
async def predict(person: Census):
    data = pd.DataFrame.from_dict(person)
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    X, _, _, _ = process_data(
        data,
        categorical_features=cat_features,
        label=None,
        training=False,
        encoder=encoder,
        lb=lb
    )
    pred = inference(model, X)[0]
    return '<=50K' if pred == 0 else '>50K'

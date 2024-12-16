# Put the code for your API here.

import os
import pickle
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field, ConfigDict
# from pydantic.alias_generators import to_snake, to_pascal
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
    return string.replace('-', '_')


def field_title(string: str):
    return string.replace('_', '-')


class Census(BaseModel):
    # Using the first row of census.csv as sample
    age: int = Field(None, example=40)
    workclass: str = Field(None, example='State-gov')
    fnlgt: int = Field(None, example=88516)
    education: str = Field(None, example='Bachelors')
    education_num: int = Field(None, example=13)
    marital_status: str = Field(None,
                                example='Never-married')
    occupation: str = Field(None, example='Adm-clerical')
    relationship: str = Field(None, example='Not-in-family')
    race: str = Field(None, example='Black')
    sex: str = Field(None, example='Female')
    capital_gain: int = Field(None, example=2174)
    capital_loss: int = Field(None, example=0)
    hours_per_week: int = Field(None, example=40)
    native_country: str = Field(None, example='Peru')

    model_config = ConfigDict(alias_generator=field_title)


@app.get("/")
async def home_page():
    return "Welcome to the Census income class predictor API!"


@app.post("/")
async def predict(data: Census):
    data_shape = pd.DataFrame.from_dict(data.__dict__)
    print('data size: {}'.format(data_shape.shape))
    data = {key.replace('_', '-'): [value] for key, value in data.__dict__.items()}
    data = pd.DataFrame.from_dict(data)
    print('data size: {}'.format(data.shape))


    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country"
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

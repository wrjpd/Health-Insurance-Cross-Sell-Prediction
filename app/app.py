from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd

# Load model
with open("./models/pipeline_prod.pkl", "rb") as f:
    model = pickle.load(f)
    
# Define input schema
class PredictRequest(BaseModel):
    Gender: str
    Age: int
    Driving_License: str
    Region_Code: str
    Previously_Insured: str
    Vehicle_Age: str
    Vehicle_Damage: str
    Annual_Premium: float
    Policy_Sales_Channel: float
    Vintage: int
    
    
app = FastAPI()
@app.get("/")
def home():
    return {"message": "Welcome to the Health Insurance Prediction API!"}
@app.post("/predict")
def predict_insurance(data: PredictRequest):
    # Converte o Pydantic model para dict
    input_dict = data.dict()
    
    # Converte para DataFrame com uma linha
    features = pd.DataFrame([input_dict])
    
    # Passa para o pipeline
    prediction = model.predict(features)
    return {"prediction": prediction.tolist()}
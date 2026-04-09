import os
import joblib
import numpy as np

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from preprocess import preprocess_input


app = FastAPI(
    title="Salary Prediction API",
    description="API for predicting salary in USD using the trained model.",
    version="1.0.0"
)


# Build paths safely
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")

MODEL_PATH = os.path.join(MODELS_DIR, "salary_model.pkl")
COLUMNS_PATH = os.path.join(MODELS_DIR, "model_columns.pkl")
TOP_LOCATIONS_PATH = os.path.join(MODELS_DIR, "top_locations.pkl")
EXPERIENCE_MAP_PATH = os.path.join(MODELS_DIR, "experience_map.pkl")
SIZE_MAP_PATH = os.path.join(MODELS_DIR, "size_map.pkl")


# Load artifacts once when API starts
try:
    model = joblib.load(MODEL_PATH)
    model_columns = joblib.load(COLUMNS_PATH)
    top_locations = joblib.load(TOP_LOCATIONS_PATH)
    experience_map = joblib.load(EXPERIENCE_MAP_PATH)
    size_map = joblib.load(SIZE_MAP_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load model artifacts: {e}")


class SalaryInput(BaseModel):
    experience_level: str = Field(..., example="SE")
    employment_type: str = Field(..., example="FT")
    company_size: str = Field(..., example="M")
    remote_ratio: int = Field(..., example=100)
    company_location: str = Field(..., example="US")
    job_title: str = Field(..., example="Data Scientist")


@app.get("/")
def root():
    return {"message": "Salary Prediction API is running"}


@app.post("/predict")
def predict_salary(data: SalaryInput):
    try:
        input_dict = data.model_dump()

        processed_input = preprocess_input(
            input_dict=input_dict,
            model_columns=model_columns,
            top_locations=top_locations,
            experience_map=experience_map,
            size_map=size_map
        )

        pred_log = model.predict(processed_input)[0]
        pred_salary = float(np.expm1(pred_log))

        return {
            "input": input_dict,
            "predicted_salary_usd": round(pred_salary, 2)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
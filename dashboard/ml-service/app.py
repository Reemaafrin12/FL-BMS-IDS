"""FastAPI inference service for the centralized EVSE Random Forest model."""

from __future__ import annotations

import os
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


MODEL_PATH = Path(os.getenv("MODEL_PATH", Path(__file__).resolve().parents[2] / "model.pkl"))
FEATURE_COLUMNS = ["shunt_voltage", "bus_voltage_V", "current_mA", "power_mW", "State"]

app = FastAPI(title="EVSE IDS Prediction Service")
model = None


class Reading(BaseModel):
    shunt_voltage: float
    bus_voltage_V: float
    current_mA: float
    power_mW: float
    State: str = Field(min_length=1)


@app.on_event("startup")
def load_model() -> None:
    global model
    if not MODEL_PATH.is_file():
        raise RuntimeError(f"Model file not found: {MODEL_PATH}. Run train_baseline.py --save-model model.pkl first.")
    model = joblib.load(MODEL_PATH)
    model.n_jobs = 1


def encode_reading(reading: Reading) -> pd.DataFrame:
    raw = pd.DataFrame([reading.model_dump()], columns=FEATURE_COLUMNS)
    encoded = pd.get_dummies(raw, columns=["State"], dtype=int)
    # The saved Random Forest records the DataFrame columns it was trained on.
    return encoded.reindex(columns=model.feature_names_in_, fill_value=0)


@app.post("/predict")
def predict(reading: Reading) -> dict[str, float | str]:
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    features = encode_reading(reading)
    prediction = str(model.predict(features)[0])
    probability_index = list(model.classes_).index(prediction)
    confidence = float(model.predict_proba(features)[0][probability_index])
    return {"prediction": prediction, "confidence": confidence}


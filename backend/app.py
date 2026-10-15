"""FastAPI backend. Run from the backend folder:  uvicorn app:app --reload"""
import json
from pathlib import Path
from typing import List

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import features  # noqa: F401  (needed so joblib can unpickle the model)

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "model.joblib"
if not MODEL_PATH.exists():
    raise RuntimeError("model.joblib not found. Run `python train_model.py` from the project root first.")

bundle = joblib.load(MODEL_PATH)
pipeline, THRESHOLD, MODEL_NAME = bundle["pipeline"], bundle["threshold"], bundle["model_name"]
METRICS = json.loads((HERE / "metrics.json").read_text()) if (HERE / "metrics.json").exists() else {}

app = FastAPI(title="SMS Spam Classifier API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class Message(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000)


class Batch(BaseModel):
    messages: List[str] = Field(..., min_length=1, max_length=500)


def _result(text: str, prob: float):
    return {"message": text, "label": "spam" if prob >= THRESHOLD else "ham",
            "spam_probability": round(float(prob), 4), "threshold": round(THRESHOLD, 4)}


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME}


@app.get("/model-info")
def model_info():
    return {"model": MODEL_NAME, "threshold": THRESHOLD, "metrics": METRICS}


@app.post("/predict")
def predict(body: Message):
    prob = pipeline.predict_proba([body.message])[0, 1]
    return _result(body.message, prob)


@app.post("/predict_batch")
def predict_batch(body: Batch):
    if any(not isinstance(m, str) for m in body.messages):
        raise HTTPException(400, "All messages must be strings")
    probs = pipeline.predict_proba(body.messages)[:, 1]
    return {"results": [_result(m, p) for m, p in zip(body.messages, probs)]}
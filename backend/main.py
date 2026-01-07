from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

import joblib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


APP_DIR = Path(__file__).resolve().parent

MODEL_PATH = Path(os.getenv("MODEL_PATH", APP_DIR / "data" / "06_models" / "toxicity_model.pkl"))
THRESH_PATH = Path(os.getenv("THRESH_PATH", APP_DIR / "data" / "06_models" / "toxicity_threshold.json"))

app = FastAPI(title="Toxic or Not API", version="0.1.0")

# For GitHub Pages -> Render calls. Tighten later to your GitHub Pages origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://simoneassirelli.github.io"
        ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    toxicity_score: float
    is_toxic: bool
    threshold: float


def _load_threshold(default: float = 0.5) -> float:
    if THRESH_PATH.exists():
        try:
            data = json.loads(THRESH_PATH.read_text(encoding="utf-8"))
            return float(data.get("threshold", default))
        except Exception:
            return default
    return default


# Load artifacts at startup
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Run your modeling pipeline first.")

model = joblib.load(MODEL_PATH)
threshold = _load_threshold(0.5)


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "threshold": threshold}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    proba = float(model.predict_proba([req.text])[0][1])
    return PredictResponse(toxicity_score=proba, is_toxic=proba >= threshold, threshold=threshold)

@app.get("/")
def root():
    return {"message": "Toxic or Not API. Use /health or POST /predict"}

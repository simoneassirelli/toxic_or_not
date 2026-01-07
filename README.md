# Toxic or Not

A small end-to-end machine learning project that detects whether a text
comment is **toxic** or **not toxic**.

-   **Pipelines:** Kedro (data ingestion → exploration → training →
    evaluation)
-   **Backend:** FastAPI (served on Render)
-   **Frontend:** Static HTML/JS (served on GitHub Pages)

## What it does

1.  Loads the Jigsaw Toxic Comment dataset (manually downloaded from
    Kaggle).
2.  Trains a baseline model (TF-IDF + Logistic Regression).
3.  Exposes a `/predict` API endpoint returning a toxicity probability
    and a boolean decision.
4.  Provides a simple web UI to type a comment and get a result.

## Repository structure

    frontend/   # GitHub Pages static site (HTML/CSS/JS)
    backend/    # Kedro project + FastAPI API

## Local setup

### 1) Create & activate a virtual environment

Windows (PowerShell):

``` powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 2) Install dependencies

``` powershell
pip install -r backend/requirements.txt
```

## Data (manual download)

This project uses the Kaggle **Jigsaw Toxic Comment Classification
Challenge** dataset.

1.  Download `train.csv` from Kaggle:
    https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data
2.  Place it here:

```{=html}
<!-- -->
```
    backend/data/01_raw/jigsaw_toxic_comment/train.csv

## Run Kedro pipelines

From `backend/`:

``` powershell
python -m kedro run --pipeline data_engineering
python -m kedro run --pipeline data_exploration
python -m kedro run --pipeline modeling
python -m kedro run --pipeline inference
```

Artifacts are written under: - `backend/data/06_models/` (model +
threshold) - `backend/data/08_reporting/` (metrics/reporting)

## Run the API locally

From `backend/`:

``` powershell
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Test:

``` powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/health"
```

## Run the UI locally

From `frontend/`:

``` powershell
python -m http.server 5173
```

Open: http://127.0.0.1:5173

Set the API URL to: http://127.0.0.1:8000

## Deployment

-   **Backend:** Render (FastAPI)
-   **Frontend:** GitHub Pages

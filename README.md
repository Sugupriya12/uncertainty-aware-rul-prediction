# Uncertainty-Aware Jet Engine Remaining Useful Life Prediction

## Overview
A machine learning project that predicts the Remaining Useful Life (RUL) of jet engines using the PHM 2008 Prognostics Challenge dataset. Alongside point predictions, the system provides uncertainty intervals to communicate prediction uncertainty.

## Features
- Random Forest model for RUL prediction.
- Quantile regression models for uncertainty intervals.
- FastAPI backend for serving predictions.
- Interactive HTML dashboard for submitting sensor readings.
- Input validation for missing and non-finite values.
- Automated API tests using pytest.

## Tech Stack
Python, pandas, NumPy, scikit-learn, PyTorch, FastAPI, Pydantic, HTML, CSS, JavaScript, pytest.

## Project Structure
- `app/` — API and dashboard
- `models/` — trained models and configuration
- `data/` — dataset and processed data
- `notebooks/` — experimentation and model development
- `src/` — project source code
- `tests/` — API tests
- `results/` — evaluation outputs
- `configs/` — project configuration

## Setup

Create and activate a virtual environment, then install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run the API

From the project root:

```powershell
python -m uvicorn app.main:app --reload
```

- Dashboard: http://127.0.0.1:8000/dashboard
- API documentation: http://127.0.0.1:8000/docs
- Health check: http://127.0.0.1:8000/health

## Run Tests

```powershell
python -m pytest -v
```

## Model Evaluation
The tuned Random Forest achieved a validation MAE of approximately **26.27 cycles** and RMSE of approximately **35.05 cycles**. The uncertainty interval is an estimate and does not guarantee coverage for every individual engine.

## Limitations
Predictions depend on the training data and feature distribution. Model performance may vary across engine conditions and RUL ranges. This project is for research and demonstration, not operational aviation maintenance decisions.

## Dataset
This project uses the **PHM 2008 Prognostics Challenge dataset** for turbofan engine remaining useful life prediction. Refer to the original challenge documentation for dataset details and applicable usage terms.

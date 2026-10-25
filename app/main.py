from fastapi.responses import HTMLResponse
from pathlib import Path
import json
from pathlib import Path
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

# Load saved models and configuration
try:
    rf_model = joblib.load(MODELS_DIR / "rul_random_forest_tuned.joblib")
    lower_model = joblib.load(MODELS_DIR / "rul_lower_quantile.joblib")
    upper_model = joblib.load(MODELS_DIR / "rul_upper_quantile.joblib")

    with open(MODELS_DIR / "uncertainty_config.json", "r") as file:
        config = json.load(file)

    FEATURE_COLUMNS = config["feature_columns"]

except Exception as exc:
    raise RuntimeError(f"Could not load model artifacts: {exc}") from exc


app = FastAPI(
    title="Jet Engine Remaining Useful Life API",
    description="Predicts remaining useful life (RUL) with uncertainty bounds.",
    version="1.0.0",
)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(
        status_code=422,
        content={"detail": "Invalid input. All sensor values must be finite numbers."},
    )

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    html_path = Path(__file__).parent / "dashboard.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))

class SensorInput(BaseModel):
    setting_1: float = Field(..., allow_inf_nan=False)
    setting_2: float = Field(..., allow_inf_nan=False)
    setting_3: float = Field(..., allow_inf_nan=False)

    sensor_1: float = Field(..., allow_inf_nan=False)
    sensor_2: float = Field(..., allow_inf_nan=False)
    sensor_3: float = Field(..., allow_inf_nan=False)
    sensor_4: float = Field(..., allow_inf_nan=False)
    sensor_5: float = Field(..., allow_inf_nan=False)
    sensor_6: float = Field(..., allow_inf_nan=False)
    sensor_7: float = Field(..., allow_inf_nan=False)
    sensor_8: float = Field(..., allow_inf_nan=False)
    sensor_9: float = Field(..., allow_inf_nan=False)
    sensor_10: float = Field(..., allow_inf_nan=False)
    sensor_11: float = Field(..., allow_inf_nan=False)
    sensor_12: float = Field(..., allow_inf_nan=False)
    sensor_13: float = Field(..., allow_inf_nan=False)
    sensor_14: float = Field(..., allow_inf_nan=False)
    sensor_15: float = Field(..., allow_inf_nan=False)
    sensor_16: float = Field(..., allow_inf_nan=False)
    sensor_17: float = Field(..., allow_inf_nan=False)
    sensor_18: float = Field(..., allow_inf_nan=False)
    sensor_19: float = Field(..., allow_inf_nan=False)
    sensor_20: float = Field(..., allow_inf_nan=False)
    sensor_21: float = Field(..., allow_inf_nan=False)


@app.get("/")
def home():
    return {"message": "Jet Engine RUL Prediction API is running"}


@app.get("/health")
def health():
    return {"status": "healthy", "models_loaded": True}


@app.post("/predict")
def predict_rul(data: SensorInput):
    try:
        input_data = pd.DataFrame(
            [data.model_dump()],
            columns=FEATURE_COLUMNS,
        )

        prediction = float(rf_model.predict(input_data)[0])
        lower = float(lower_model.predict(input_data)[0])
        upper = float(upper_model.predict(input_data)[0])

        adjustment = float(config.get("conformal_adjustment", 0.0))
        lower -= adjustment
        upper += adjustment

        # RUL cannot be negative; keep the interval valid.
        prediction = max(0.0, prediction)
        lower = max(0.0, lower)
        upper = max(lower, upper)

        return {
            "predicted_rul_cycles": round(prediction, 2),
            "uncertainty_interval_cycles": {
                "lower": round(lower, 2),
                "upper": round(upper, 2),
            },
            "interval_target_coverage": config.get("target_coverage"),
            "note": (
                "The interval is an estimate, not a guarantee for an "
                "individual engine. Model performance varies across RUL ranges."
            ),
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Check the server logs.",
        ) from exc

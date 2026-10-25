# Uncertainty-Aware Jet Engine Remaining Useful Life Prediction

A machine learning project that predicts the remaining useful life (RUL) of jet engines from operating settings and sensor measurements. It combines a tuned Random Forest regressor, quantile-based prediction intervals, a FastAPI service, and an interactive dashboard.

## Problem Statement

Engine degradation is difficult to predict from sensor measurements alone. A point estimate of remaining useful life does not communicate how uncertain that estimate may be.

This project predicts RUL in cycles and provides an estimated prediction interval to make uncertainty more visible when interpreting engine health predictions.

## Features

- Predicts RUL using 3 operating settings and 21 sensor measurements.
- Compares baseline and tuned Random Forest regression models.
- Estimates prediction intervals using quantile models and a calibration adjustment.
- Uses an engine-level training/validation split to avoid engine overlap between the two sets.
- Exposes predictions through a FastAPI REST API.
- Includes an interactive web dashboard.
- Tests valid predictions, health checks, missing inputs, and non-finite values.

## Validation Results

| Model / metric | Result |
|---|---:|
| Baseline Random Forest MAE | 26.801 cycles |
| Baseline Random Forest RMSE | 35.695 cycles |
| Tuned Random Forest MAE | 26.274 cycles |
| Tuned Random Forest RMSE | 35.047 cycles |
| Prediction interval coverage before adjustment | 78.09% |
| Prediction interval coverage after adjustment | 78.91% |
| Target interval coverage | 80% |
| Mean prediction interval width | 107.77 cycles |
| Calibration adjustment | 0.8744 cycles |

The tuned Random Forest delivers a **modest improvement** over the baseline Random Forest: MAE decreases by approximately 1.97%, while RMSE decreases by approximately 1.81%.

The calibrated prediction intervals cover 78.91% of the evaluated observations, compared with the 80% target. The mean interval width is 107.77 cycles, indicating that the uncertainty ranges are broad. This is an important limitation when interpreting individual predictions.

The target coverage is a calibration objective, not a guarantee for every future prediction. These results describe the project's validation evaluation, not a verified score on the challenge's official test set.

**Baseline note:** The table compares two Random Forest models. A constant mean-RUL predictor is a separate, simpler baseline and is not included because its validation metrics have not yet been calculated. It should be evaluated on the same validation split before making claims about the improvement over a trivial predictor.

## Dataset

The project uses the **PHM 2008 Prognostics Challenge turbofan engine degradation dataset**, commonly distributed with `train.txt`, `test.txt`, and a separate RUL-label file.

The training data contains run-to-failure engine histories. RUL targets are derived from the difference between an engine's final observed cycle and its current cycle. The official test setup supplies final RUL labels separately, so those labels should not be confused with the project's own validation targets.

### Engine-level split

| Split | Rows | Engines |
|---|---:|---:|
| Training | 36,420 | 174 |
| Validation | 9,498 | 44 |
| Total | 45,918 | 218 |

No engine IDs overlap between the training and validation splits.

The project reports results on its held-out validation split. The preprocessing and evaluation details should be consulted when comparing these numbers with results from other implementations, especially if they use a different RUL target transformation.

### Dataset reference

A. Saxena and K. Goebel, “Turbofan Engine Degradation Simulation Data Set,” NASA Ames Prognostics Data Repository, 2008. See the [NASA Prognostics Data Repository](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/).

## Methodology

1. **Data preparation:** Process operating settings and sensor measurements and create RUL targets.
2. **Engine-level splitting:** Separate training and validation engines to prevent engine overlap.
3. **Baseline model:** Train a Random Forest regression model for comparison.
4. **Model tuning:** Evaluate a tuned Random Forest using MAE and RMSE.
5. **Uncertainty estimation:** Use quantile models and a calibration adjustment to construct prediction intervals.
6. **API deployment:** Serve RUL predictions and interval estimates through FastAPI.
7. **Testing:** Validate API behavior and input handling.

## Technology Stack

- **Language:** Python
- **Data processing:** pandas, NumPy
- **Machine learning:** scikit-learn
- **Experimental deep learning:** PyTorch
- **API:** FastAPI, Uvicorn
- **Validation and testing:** Pydantic, pytest, HTTPX
- **Interface:** HTML, CSS, JavaScript

The deployed API uses the tuned Random Forest and saved uncertainty-estimation artifacts. The saved PyTorch CNN (`rul_cnn.pt`) is an experimental model artifact; it is not used by the current prediction API, and no CNN performance claim is made here.

## Project Structure

```text
uncertainty-aware-rul-prediction/
├── app/
│   ├── main.py
│   └── dashboard.html
├── data/
│   └── processed/
│       ├── train_split.csv
│       ├── validation_split.csv
│       └── split_metadata.json
├── models/
│   ├── rul_random_forest_tuned.joblib
│   ├── rul_lower_quantile.joblib
│   ├── rul_upper_quantile.joblib
│   ├── uncertainty_config.json
│   └── rul_cnn.pt
├── notebooks/
├── results/
│   └── baseline_results.csv
├── src/
│   └── inspect_data.py
├── tests/
│   └── test_api.py
├── requirements.txt
├── .gitignore
└── README.md
```

### Training and retraining

The exploratory training and evaluation workflow lives in the Jupyter notebooks under `notebooks/`. The `src/` directory currently contains data-inspection code rather than a standalone end-to-end training script.

The API loads saved model artifacts from `models/`. Retraining requires running the relevant notebook workflow and updating the artifacts expected by the application.

## Setup and Installation

### 1. Clone the repository

```bash
git clone https://github.com/Sugupriya12/uncertainty-aware-rul-prediction.git
cd uncertainty-aware-rul-prediction
```

### 2. Create a virtual environment

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

## Run the Application

Start the FastAPI server:

```bash
python -m uvicorn app.main:app --reload
```

Open the following URLs:

- Dashboard: http://127.0.0.1:8000/dashboard
- API documentation: http://127.0.0.1:8000/docs
- Health check: http://127.0.0.1:8000/health

The application is configured for local use. Public deployment would require appropriate security and deployment configuration.

## API Usage

The prediction endpoint is `POST /predict`. It accepts 24 numerical inputs: `setting_1`, `setting_2`, `setting_3`, and `sensor_1` through `sensor_21`.

### Example request structure

```json
{
  "setting_1": 0.0,
  "setting_2": 0.0,
  "setting_3": 0.0,
  "sensor_1": 0.0,
  "sensor_2": 0.0,
  "sensor_3": 0.0,
  "sensor_4": 0.0,
  "sensor_5": 0.0,
  "sensor_6": 0.0,
  "sensor_7": 0.0,
  "sensor_8": 0.0,
  "sensor_9": 0.0,
  "sensor_10": 0.0,
  "sensor_11": 0.0,
  "sensor_12": 0.0,
  "sensor_13": 0.0,
  "sensor_14": 0.0,
  "sensor_15": 0.0,
  "sensor_16": 0.0,
  "sensor_17": 0.0,
  "sensor_18": 0.0,
  "sensor_19": 0.0,
  "sensor_20": 0.0,
  "sensor_21": 0.0
}
```

The zero values are placeholders illustrating the input schema, not realistic engine measurements.

### Example response

```json
{
  "predicted_rul_cycles": 159.31,
  "uncertainty_interval_cycles": {
    "lower": 86.44,
    "upper": 222.97
  },
  "interval_target_coverage": 0.8,
  "note": "The interval is an estimate, not a guarantee for an individual prediction."
}
```

This example shows a prediction of 159.31 cycles and an estimated interval from 86.44 to 222.97 cycles. It is one API example, not a summary of model performance.

## Tests

Run the test suite with:

```bash
python -m pytest -q
```

The test suite contains four API tests covering the health endpoint, a valid prediction and interval, missing sensor input, and non-finite input values.

## Limitations and Future Work

- Add and evaluate a constant mean-RUL baseline on the same validation split.
- Investigate the broad prediction intervals and the gap between empirical and target coverage.
- Document the exact RUL target transformation used during preprocessing.
- Evaluate the experimental CNN before making performance comparisons.
- Add a dashboard screenshot to make the repository easier to explore.
- Consider a standalone training script to simplify reproducibility.

This project is intended for machine learning experimentation and demonstration. It is not a certified aviation maintenance system.

## License

No explicit open-source license has been added to this repository yet. Dataset access and redistribution are subject to the original dataset's terms. A repository license should be selected intentionally before presenting the project as open source.

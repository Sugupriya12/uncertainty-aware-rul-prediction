
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app, raise_server_exceptions=False)

SAMPLE_INPUT = {
    "setting_1": 0.0023,
    "setting_2": 0.0002,
    "setting_3": 100.0,
    "sensor_1": 518.67,
    "sensor_2": 642.0,
    "sensor_3": 1580.0,
    "sensor_4": 1400.0,
    "sensor_5": 14.62,
    "sensor_6": 21.61,
    "sensor_7": 554.36,
    "sensor_8": 2388.0,
    "sensor_9": 9046.0,
    "sensor_10": 1.3,
    "sensor_11": 47.47,
    "sensor_12": 521.66,
    "sensor_13": 2388.0,
    "sensor_14": 8138.62,
    "sensor_15": 8.42,
    "sensor_16": 0.03,
    "sensor_17": 392.0,
    "sensor_18": 2388.0,
    "sensor_19": 100.0,
    "sensor_20": 39.0,
    "sensor_21": 23.4,
}


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["models_loaded"] is True


def test_prediction_returns_valid_interval():
    response = client.post("/predict", json=SAMPLE_INPUT)

    assert response.status_code == 200
    result = response.json()

    assert result["predicted_rul_cycles"] >= 0
    interval = result["uncertainty_interval_cycles"]
    assert interval["lower"] >= 0
    assert interval["upper"] >= interval["lower"]
    assert result["interval_target_coverage"] == 0.8


def test_missing_sensor_is_rejected():
    incomplete_input = SAMPLE_INPUT.copy()
    del incomplete_input["sensor_21"]

    response = client.post("/predict", json=incomplete_input)
    assert response.status_code == 422





def test_non_finite_input_is_rejected():
    import json

    invalid_input = SAMPLE_INPUT.copy()
    invalid_input["sensor_1"] = float("inf")

    response = client.post(
        "/predict",
        content=json.dumps(invalid_input, allow_nan=True),
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 422


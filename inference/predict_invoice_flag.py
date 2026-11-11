import joblib
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "predict_flag_invoice.pkl"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"


def load_model():
    return joblib.load(MODEL_PATH)


def load_scaler():
    return joblib.load(SCALER_PATH)


def predict_invoice_flag(input_data):

    model = load_model()
    scaler = load_scaler()

    input_df = pd.DataFrame(input_data)

    features = [
        "invoice_quantity",
        "invoice_dollars",
        "Freight",
        "total_item_quantity",
        "total_item_dollars"
    ]

    model_input = input_df[features]
    model_input_scaled = scaler.transform(model_input)

    predictions = model.predict(model_input_scaled)

    input_df["Predicted_Flag"] = predictions.astype(int)

    return input_df
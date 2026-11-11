import joblib
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

model_path = BASE_DIR / "models" / "predict_freight_model.pkl"


def load_model():
    return joblib.load(model_path)


def predict_freight_model(input_data):

    model = load_model()

    input_df = input_data.copy()

    expected_features = model.feature_names_in_

    input_df = input_df.reindex(columns=expected_features)

    prediction = model.predict(input_df).round(2)

    return prediction[0]


if __name__ == '__main__':

    sample_data = pd.DataFrame({
        "Quantity": [1200, 500, 200, 50],
        "Dollars": [18500, 9000, 3000, 200]
    })

    prediction = predict_freight_model(sample_data)

    print(prediction)
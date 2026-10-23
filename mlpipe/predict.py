
from pathlib import Path

import joblib
import pandas as pd


def predict(
    input_path="data/new_patients.csv",
    output_path="reports/predictions.csv"
):
    model_path = Path("models/model.joblib")

    if not model_path.exists():
        raise FileNotFoundError(
            "Модель не найдена. Сначала запусти train."
        )

    artifact = joblib.load(model_path)
    pipeline = artifact["pipeline"]
    features = artifact["features"]

    df = pd.read_csv(input_path)

    if df.empty:
        raise ValueError("Входной CSV пустой")

    missing = set(features) - set(df.columns)

    if missing:
        raise ValueError(
            f"Отсутствуют признаки: {sorted(missing)}"
        )

    X = df[features].copy()
    predictions = pipeline.predict(X)

    result = df.copy()
    result["prediction"] = predictions
    result["prediction_label"] = [
        "Benign" if value == 1 else "Malignant"
        for value in predictions
    ]

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    result.to_csv(output, index=False)

    print("Predictions completed!")
    print(result["prediction_label"].value_counts())
    print(f"Saved to: {output}")

    return result

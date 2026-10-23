
import pytest

from mlpipe.config import TARGET
from mlpipe.data import load_data
from mlpipe.features import build_preprocessor
from mlpipe.models import get_models
from mlpipe.predict import predict


def test_data_loading():
    df = load_data()

    assert len(df) == 569
    assert TARGET in df.columns
    assert df.shape[1] == 31
    assert df.duplicated().sum() == 0


def test_preprocessor():
    df = load_data()
    X = df.drop(columns=[TARGET])

    preprocessor = build_preprocessor(X)
    transformed = preprocessor.fit_transform(X)

    assert transformed.shape == (569, 30)


def test_models():
    models = get_models()

    assert len(models) == 5
    assert "LogisticRegression" in models
    assert "Dummy" in models


def test_missing_data(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_data(tmp_path / "missing.csv")


def test_prediction(tmp_path):
    df = load_data()

    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.csv"

    df.drop(columns=[TARGET]).head(3).to_csv(
        input_path,
        index=False
    )

    result = predict(input_path, output_path)

    assert len(result) == 3
    assert "prediction" in result.columns
    assert "prediction_label" in result.columns
    assert output_path.exists()

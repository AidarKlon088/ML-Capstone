
import json
from pathlib import Path

import joblib
from sklearn.base import clone
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score
)
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from mlpipe.config import SEED, TARGET, MODEL, METRICS
from mlpipe.data import load_data
from mlpipe.features import build_preprocessor
from mlpipe.models import get_models


def train():
    # 1. Загрузка данных
    df = load_data()

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # 2. Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=SEED,
        stratify=y
    )

    # 3. Cross-Validation
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED
    )

    results = {}
    best_score = -1
    best_name = None
    best_pipeline = None
    dummy_score = None

    print("=== MODEL TRAINING ===")

    # 4. Обучение и сравнение моделей
    for name, model in get_models().items():

        pipeline = Pipeline([
            ("preprocessor", build_preprocessor(X_train)),
            ("model", clone(model))
        ])

        scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring="roc_auc",
            n_jobs=-1
        )

        mean_score = float(scores.mean())

        results[name] = {
            "cv_roc_auc_mean": mean_score,
            "cv_roc_auc_std": float(scores.std())
        }

        print(f"{name}: ROC-AUC = {mean_score:.4f}")

        if name == "Dummy":
            dummy_score = mean_score
            continue

        if mean_score > best_score:
            best_score = mean_score
            best_name = name
            best_pipeline = pipeline

    # 5. Baseline-gate
    if dummy_score is None:
        raise ValueError("DummyClassifier отсутствует")

    improvement = best_score - dummy_score

    print("\n=== BASELINE CHECK ===")
    print(f"Dummy ROC-AUC: {dummy_score:.4f}")
    print(f"Best ROC-AUC: {best_score:.4f}")
    print(f"Improvement: {improvement:.4f}")

    if improvement < 0.05:
        raise RuntimeError(
            "Модель не прошла baseline-gate!"
        )

    print("Baseline-gate PASSED!")

    # 6. Обучение лучшей модели
    print(f"\nBest model: {best_name}")

    best_pipeline.fit(X_train, y_train)

    # 7. Проверка на тестовой выборке
    predictions = best_pipeline.predict(X_test)

    if hasattr(best_pipeline, "predict_proba"):
        probabilities = best_pipeline.predict_proba(X_test)[:, 1]
    else:
        probabilities = best_pipeline.decision_function(X_test)

    test_metrics = {
        "accuracy": float(
            accuracy_score(y_test, predictions)
        ),
        "precision": float(
            precision_score(y_test, predictions)
        ),
        "recall": float(
            recall_score(y_test, predictions)
        ),
        "f1": float(
            f1_score(y_test, predictions)
        ),
        "roc_auc": float(
            roc_auc_score(y_test, probabilities)
        ),
        "confusion_matrix": confusion_matrix(
            y_test, predictions
        ).tolist()
    }

    # 8. Создание папок
    Path(MODEL).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    Path(METRICS).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # 9. Сохранение модели
    joblib.dump(
        {
            "pipeline": best_pipeline,
            "features": X.columns.tolist()
        },
        MODEL
    )

    # 10. Сохранение метрик
    with open(
        METRICS,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            {
                "best_model": best_name,
                "baseline_roc_auc": dummy_score,
                "improvement": improvement,
                "cross_validation": results,
                "test_metrics": test_metrics
            },
            file,
            indent=4
        )

    print("\n=== TEST METRICS ===")
    print(json.dumps(test_metrics, indent=4))

    print("\nModel saved successfully!")
    print(f"Model: {MODEL}")
    print(f"Metrics: {METRICS}")

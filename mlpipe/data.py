import pandas as pd

from mlpipe.config import TARGET, DATA


def load_data(path=DATA):
    df = pd.read_csv(path)

    if TARGET not in df.columns:
        raise ValueError(
            f"Не найдена целевая колонка: {TARGET}"
        )

    if df.empty:
        raise ValueError("Датасет пустой")

    duplicates = df.duplicated().sum()

    if duplicates:
        print(f"Удалено дубликатов: {duplicates}")
        df = df.drop_duplicates()

    return df


def run_eda(path=DATA):
    df = load_data(path)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    print("=== EDA REPORT ===")

    print(f"Строк: {len(X)}")
    print(f"Признаков: {X.shape[1]}")

    print("Пропусков:", X.isna().sum().sum())

    print("\nБаланс классов:")
    print(y.value_counts(normalize=True).round(3))

    print("\nТоп-3 корреляции:")

    correlations = (
        X.corrwith(y)
        .abs()
        .sort_values(ascending=False)
        .head(3)
    )

    print(correlations.round(2))
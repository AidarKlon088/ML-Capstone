from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)


def build_preprocessor(X=None):

    if X is None:
        raise ValueError(
            "Передайте DataFrame X для определения типов признаков"
        )

    num = X.select_dtypes(include="number").columns.tolist()

    cat = X.columns.difference(num).tolist()

    return ColumnTransformer([
        (
            "num",
            make_pipeline(
                SimpleImputer(strategy="median"),
                StandardScaler()
            ),
            num
        ),

        (
            "cat",
            make_pipeline(
                SimpleImputer(strategy="most_frequent"),
                OneHotEncoder(handle_unknown="ignore")
            ),
            cat
        )
    ])
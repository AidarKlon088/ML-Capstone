
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


def get_models():
    return {
        "Dummy": DummyClassifier(strategy="most_frequent"),

        "LogisticRegression": LogisticRegression(
            max_iter=2000,
            random_state=42
        ),

        "RandomForest": RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ),

        "SVM": SVC(
            kernel="rbf",
            probability=True,
            random_state=42
        ),

        "KNN": KNeighborsClassifier(
            n_neighbors=5
        )
    }

from pathlib import Path
from sklearn.datasets import load_breast_cancer

Path("data").mkdir(exist_ok=True)

df = load_breast_cancer(as_frame=True).frame

df["malignant"] = 1 - df.pop("target")

df.to_csv("data/raw.csv", index=False)

new = df.sample(5, random_state=1).drop(columns="malignant")
new.to_csv("data/new_patients.csv", index=False)

print("Данные успешно созданы!")
print("Размер:", df.shape)
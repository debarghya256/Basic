"""Train an Iris flower classifier and save it to disk.

Run:  python train.py
"""
import json
from pathlib import Path

import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

MODEL_DIR = Path("model")
MODEL_DIR.mkdir(exist_ok=True)


def main():
    # 1. Load data (built into scikit-learn, no download needed)
    iris = load_iris()
    X, y = iris.data, iris.target

    # 2. Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Train a model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 4. Evaluate on data the model has never seen
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Test accuracy: {acc:.3f}\n")
    print(classification_report(y_test, preds, target_names=iris.target_names))

    # 5. Save the model + metadata so the API can load it
    joblib.dump(model, MODEL_DIR / "iris_model.joblib")
    metadata = {
        "feature_names": list(iris.feature_names),
        "class_names": list(iris.target_names),
        "test_accuracy": round(acc, 4),
    }
    (MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Saved model to {MODEL_DIR}/")


if __name__ == "__main__":
    main()

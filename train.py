"""Train and evaluate the SignalCheck headline classifier."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.request import urlretrieve

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model.joblib"
METRICS_PATH = ROOT / "metrics.json"
BASE_URL = "https://raw.githubusercontent.com/KaiDMML/FakeNewsNet/master/dataset"
FILES = {
    "politifact_fake.csv": 0,
    "politifact_real.csv": 1,
    "gossipcop_fake.csv": 0,
    "gossipcop_real.csv": 1,
}


def load_dataset(data_dir: Path = ROOT) -> pd.DataFrame:
    """Download the official FakeNewsNet title files and return clean labeled rows."""
    frames = []
    for filename, label in FILES.items():
        path = data_dir / filename
        if not path.exists():
            urlretrieve(f"{BASE_URL}/{filename}", path)
        frame = pd.read_csv(path, usecols=["title"])
        frame["label"] = label
        frames.append(frame)

    data = pd.concat(frames, ignore_index=True)
    data["title"] = data["title"].fillna("").astype(str).str.strip()
    return data[data["title"].str.len() >= 12].drop_duplicates("title").reset_index(drop=True)


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.98,
                    max_features=45_000,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1_500,
                    class_weight="balanced",
                    C=2.0,
                    random_state=42,
                ),
            ),
        ]
    )


def train_and_save(root: Path = ROOT) -> tuple[Pipeline, dict]:
    data = load_dataset(root)
    x_train, x_test, y_train, y_test = train_test_split(
        data["title"], data["label"], test_size=0.2, stratify=data["label"], random_state=42
    )
    model = build_pipeline()
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    metrics = {
        "records": int(len(data)),
        "train_records": int(len(x_train)),
        "test_records": int(len(x_test)),
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "f1_macro": round(float(f1_score(y_test, predictions, average="macro")), 4),
        "fake_precision": round(float(report["0"]["precision"]), 4),
        "fake_recall": round(float(report["0"]["recall"]), 4),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "dataset": "FakeNewsNet — PolitiFact + GossipCop titles",
    }
    joblib.dump(model, root / MODEL_PATH.name)
    (root / METRICS_PATH.name).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return model, metrics


if __name__ == "__main__":
    _, result = train_and_save()
    print(json.dumps(result, indent=2))

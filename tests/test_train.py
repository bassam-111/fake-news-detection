"""Tests for the training pipeline, the sample dataset and predict.py."""
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.train_model import (build_pipeline, evaluate_model,
                               load_data, predict, preprocess_data,
                               save_model, split_data)
from predict import predict_news

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_CSV = ROOT / "dataset" / "sample_news.csv"


@pytest.fixture(scope="module")
def sample_df():
    return pd.read_csv(SAMPLE_CSV)


def test_sample_csv_loads_with_both_labels(sample_df):
    assert list(sample_df.columns) == ["title", "text", "label"]
    assert len(sample_df) == 40
    assert set(sample_df["label"].unique()) == {"FAKE", "REAL"}
    assert sample_df["label"].value_counts().to_dict() == {"FAKE": 20, "REAL": 20}


def test_load_data_returns_none_for_missing_file():
    assert load_data(os.path.join("no-such-dir", "news.csv")) is None


def test_preprocess_maps_labels_and_combines_text():
    df = pd.DataFrame([
        {"title": "Real headline", "text": "Some real body text.", "label": "REAL"},
        {"title": "Fake headline", "text": "Some fake body text.", "label": "FAKE"},
    ])
    out = preprocess_data(df)
    assert list(out.columns) == ["content", "label"]
    assert out["label"].tolist() == [0, 1]
    assert out["content"].iloc[0] == "Real headline Some real body text."


def test_preprocess_drops_missing_and_unknown_labels():
    df = pd.DataFrame([
        {"title": "Ok", "text": "Body.", "label": "REAL"},
        {"title": None, "text": "Body.", "label": "FAKE"},
        {"title": "Weird", "text": "Body.", "label": "MAYBE"},
    ])
    out = preprocess_data(df)
    assert len(out) == 1
    assert out["label"].tolist() == [0]


def test_split_is_stratified(sample_df):
    df = preprocess_data(sample_df)
    _, _, y_train, y_test = split_data(df["content"].values, df["label"].values)
    assert abs(y_train.mean() - 0.5) < 0.01
    assert abs(y_test.mean() - 0.5) < 0.01


def test_build_pipeline_unknown_model_raises():
    with pytest.raises(ValueError, match="Unknown model type"):
        build_pipeline("bert")


def test_pipeline_trains_and_predicts_on_sample(sample_df):
    df = preprocess_data(sample_df)
    X_train, X_test, y_train, y_test = split_data(df["content"].values, df["label"].values)
    # min_df=1, as --sample training does: with only 32 training rows the
    # default min_df=5 leaves almost no features.
    pipe = build_pipeline("passive_aggressive", min_df=1)
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    assert set(preds.tolist()) <= {0, 1}
    # The sample classes use very different wording, so a trained model
    # should do far better than chance on it.
    metrics = evaluate_model(pipe, X_test, y_test)
    assert metrics["accuracy"] >= 0.75


def test_predict_returns_label_and_margin(sample_df):
    df = preprocess_data(sample_df)
    pipe = build_pipeline("passive_aggressive")
    pipe.fit(df["content"].values, df["label"].values)
    label, score = predict(pipe, "Scientists confirm drinking cola cures cancer in 48 hours")
    assert label in {"FAKE", "REAL"}
    assert isinstance(score, float)


def test_save_and_load_roundtrip(sample_df, tmp_path):
    from model import train_model as tm

    df = preprocess_data(sample_df)
    pipe = build_pipeline("naive_bayes")
    pipe.fit(df["content"].values, df["label"].values)
    tm.MODEL_DIR = str(tmp_path)
    path = save_model(pipe, "roundtrip_model.pkl")
    assert os.path.exists(path)
    import joblib
    reloaded = joblib.load(path)
    assert reloaded.predict(["City council approves budget for road repairs"])[0] in (0, 1)


class _FakeModel:
    def __init__(self, prediction=1):
        self.prediction = prediction
        self.named_steps = {"clf": self}

    def predict(self, _texts):
        return [self.prediction]

    def decision_function(self, _texts):
        return [2.0]


def test_predict_news_returns_label_and_score():
    label, score = predict_news(_FakeModel(prediction=1), "some text")
    assert label == "FAKE"
    assert score == pytest.approx(2.0)
    label, _ = predict_news(_FakeModel(prediction=0), "some text")
    assert label == "REAL"

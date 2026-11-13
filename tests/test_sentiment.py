import json
from pathlib import Path

import joblib
import pytest

from datheon_ml.sentiment import build_pipelines, clean_text, load_datheon_sample, predict_sentiment

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "task-4-nlp-sentiment"

POS = ["great product love it", "excellent quality very happy", "amazing value works perfectly", "fantastic and durable"] * 5
NEG = ["terrible product broke quickly", "awful quality very disappointed", "worst purchase waste of money", "bad and useless"] * 5


def test_clean_text_removes_html_punctuation_and_lowercases():
    out = clean_text("<br />Great PRODUCT!!! I loved it, 10/10.")
    assert "<" not in out and "!" not in out and out == out.lower()
    assert "br" not in out.split()


def test_clean_text_lemmatises_and_removes_stopwords():
    out = clean_text("The movies were being watched by the students").split()
    assert "the" not in out and "movie" in out


def test_clean_text_keeps_negations():
    assert "not" in clean_text("This is not good").split()
    assert "not" in clean_text("It wasn't good").split()


def test_clean_text_handles_bad_input():
    assert clean_text("") == "" and clean_text(None) == ""


def test_datheon_sample_is_binary_and_complete():
    df = load_datheon_sample(TASK / "data/sentiment_reviews.csv")
    assert set(df["sentiment"]) == {"positive", "negative"} and len(df) == 9


def test_pipelines_train_and_predict_on_tiny_data():
    X = [clean_text(t) for t in POS + NEG]
    y = ["positive"] * len(POS) + ["negative"] * len(NEG)
    for name, pipe in build_pipelines(42).items():
        assert pipe.steps[0][0] == "tfidf" and pipe.steps[0][1].max_features == 5000
        pipe.fit(X, y)
        assert predict_sentiment("excellent amazing great", pipe)["sentiment"] == "positive", name
        assert predict_sentiment("terrible awful worst", pipe)["sentiment"] == "negative", name


def test_predict_sentiment_returns_expected_keys_and_handles_empty():
    model = joblib.load(TASK / "models/best_sentiment_model.joblib")
    out = predict_sentiment("This product is amazing. I really love it.", model)
    assert set(out) == {"review", "sentiment", "confidence"}
    assert out["sentiment"] in {"positive", "negative"} and 0.5 <= out["confidence"] <= 1.0
    assert predict_sentiment("", model)["sentiment"] == "unknown"


def test_recorded_metrics_are_plausible():
    m = json.loads((TASK / "results/metrics.json").read_text())
    for name, vals in m["models"].items():
        assert 0.70 <= vals["Accuracy"] <= 1.0, name
    for png in ["confusion_matrix", "sentiment_distribution", "model_comparison", "top_words"]:
        assert (TASK / f"results/{png}.png").stat().st_size > 5_000

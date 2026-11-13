"""Task 4 helpers: NLP sentiment classification."""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import nltk
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Negations carry sentiment ("not good"), so they are kept even though NLTK lists them as stopwords.
KEEP_WORDS = {"not", "no", "nor", "never", "against", "off", "very", "too"}
HTML_TAG = re.compile(r"<[^>]+>")
NON_LETTERS = re.compile(r"[^a-z\s]")


def ensure_nltk() -> None:
    """Download the NLTK resources used in this project (no-op if already present)."""
    for pkg in ("stopwords", "wordnet", "omw-1.4", "punkt", "punkt_tab", "movie_reviews"):
        nltk.download(pkg, quiet=True)


@lru_cache(maxsize=1)
def _resources():
    ensure_nltk()
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer

    stops = set(stopwords.words("english")) - KEEP_WORDS
    return stops, WordNetLemmatizer()


def clean_text(text: str) -> str:
    """HTML removal -> lowercase -> punctuation removal -> tokenise -> stopwords -> lemmatise."""
    if not isinstance(text, str):
        return ""
    stops, lemmatizer = _resources()
    text = HTML_TAG.sub(" ", text)
    text = text.lower().replace("n't", " not")
    text = NON_LETTERS.sub(" ", text)
    tokens = [lemmatizer.lemmatize(t) for t in text.split() if t not in stops and len(t) > 1]
    return " ".join(tokens)


def load_datheon_sample(path: str | Path) -> pd.DataFrame:
    """Load Datheon's supplied starter file and keep positive/negative rows only."""
    df = pd.read_csv(path).rename(columns={"review_text": "text"})
    df = df.dropna(subset=["text", "sentiment"])
    df["sentiment"] = df["sentiment"].str.strip().str.lower()
    return df[df["sentiment"].isin(["positive", "negative"])].reset_index(drop=True)


def load_movie_reviews() -> pd.DataFrame:
    """NLTK movie_reviews corpus: 2,000 labelled reviews (1,000 positive / 1,000 negative)."""
    ensure_nltk()
    from nltk.corpus import movie_reviews

    rows = [
        {"text": movie_reviews.raw(fid), "sentiment": "positive" if cat == "pos" else "negative"}
        for cat in movie_reviews.categories()
        for fid in movie_reviews.fileids(cat)
    ]
    return pd.DataFrame(rows)


def build_pipelines(random_state: int = 42, max_features: int = 5000) -> dict[str, Pipeline]:
    """TF-IDF + Multinomial Naive Bayes and TF-IDF + Logistic Regression."""
    return {
        "Multinomial Naive Bayes": Pipeline(
            [("tfidf", TfidfVectorizer(max_features=max_features)), ("model", MultinomialNB())]
        ),
        "Logistic Regression": Pipeline(
            [
                ("tfidf", TfidfVectorizer(max_features=max_features)),
                ("model", LogisticRegression(max_iter=1000, random_state=random_state)),
            ]
        ),
    }


def predict_sentiment(review: str, model: Pipeline) -> dict:
    """Clean a raw review, vectorise it with the fitted TF-IDF and predict its sentiment."""
    cleaned = clean_text(review)
    if not cleaned:
        return {"review": review, "sentiment": "unknown", "confidence": None}
    proba = model.predict_proba([cleaned])[0]
    idx = int(proba.argmax())
    return {"review": review, "sentiment": str(model.classes_[idx]), "confidence": round(float(proba[idx]), 4)}

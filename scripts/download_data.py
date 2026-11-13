"""Download the official datasets (only files that are missing) and the NLTK resources.

Usage (from the repository root):  python scripts/download_data.py
"""
from pathlib import Path
from urllib.request import urlretrieve

ROOT = Path(__file__).resolve().parents[1]
FILES = {
    ROOT / "task-1-classification/data/iris.csv":
        "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv",
    ROOT / "task-3-customer-churn/data/Telco-Customer-Churn.csv":
        "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv",
    ROOT / "task-4-nlp-sentiment/data/sentiment_reviews.csv":
        "https://www.datheon.in/datasets/sentiment_reviews.csv",
}

for path, url in FILES.items():
    if path.exists():
        print(f"[skip] {path.relative_to(ROOT)} already exists")
        continue
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        urlretrieve(url, path)
        print(f"[ok]   {path.relative_to(ROOT)}")
    except Exception as exc:  # network / link problems should be visible, never hidden
        print(f"[FAIL] {url}: {exc}\n       Contact hello@datheon.in if the Datheon link is unavailable.")

import nltk
for pkg in ("stopwords", "wordnet", "omw-1.4", "punkt", "punkt_tab", "movie_reviews"):
    nltk.download(pkg, quiet=True)
print("NLTK resources ready.")

"""Command-line sentiment predictor using the model saved by Task 4.

Usage:
  python scripts/predict_sentiment_cli.py "This product is amazing"
  python scripts/predict_sentiment_cli.py            # interactive mode
"""
import sys
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from datheon_ml.sentiment import predict_sentiment  # noqa: E402

MODEL = ROOT / "task-4-nlp-sentiment/models/best_sentiment_model.joblib"
if not MODEL.exists():
    sys.exit("Model not found. Run task-4-nlp-sentiment/sentiment_analysis.ipynb first.")
model = joblib.load(MODEL)

if len(sys.argv) > 1:
    print(predict_sentiment(" ".join(sys.argv[1:]), model))
else:
    while True:
        text = input("Review (blank to quit): ").strip()
        if not text:
            break
        print(predict_sentiment(text, model))

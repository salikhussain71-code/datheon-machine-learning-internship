"""Execute the three notebooks top-to-bottom (each inside its own task folder).

Usage (from the repository root):  python scripts/run_all_notebooks.py
Re-creates every figure in results/, every model in models/ and every metrics.json.
"""
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = [
    "task-1-classification/classification.ipynb",
    "task-3-customer-churn/customer_churn.ipynb",
    "task-4-nlp-sentiment/sentiment_analysis.ipynb",
]

for rel in NOTEBOOKS:
    path = ROOT / rel
    print(f"Running {rel} ...")
    nb = nbformat.read(path, as_version=4)
    NotebookClient(nb, timeout=3600, kernel_name="python3", resources={"metadata": {"path": str(path.parent)}}).execute()
    nbformat.write(nb, path)
    print(f"  done: {rel}")
print("All notebooks ran without errors.")

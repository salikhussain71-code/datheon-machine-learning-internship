# GitHub setup — new repository

## Repository details (copy-paste)
* **Repository name:** `datheon-machine-learning-internship`
* **Visibility:** Public
* **Description (≤ 350 chars):**
  `Datheon ML & Data Science Internship: 3 tested end-to-end projects — Iris classification, Telco customer churn (Random Forest + XGBoost, ROC-AUC 0.845), and NLP sentiment analysis (TF-IDF, Naive Bayes, Logistic Regression). Notebooks, saved models, charts, report.`
* **Topics:** `machine-learning` `data-science` `scikit-learn` `xgboost` `nlp` `sentiment-analysis` `customer-churn` `classification` `jupyter-notebook` `python` `tf-idf` `internship` `datheon`
* **Do NOT** tick "Add a README", "Add .gitignore" or "Add a license" when creating it — the project already contains them (otherwise the first push is rejected).

## Steps
1. Sign in at https://github.com → **New repository** → enter the name and description above → **Public** → leave the three "initialise" boxes unticked → **Create repository**.
2. Unzip the project, open the folder in VS Code → Terminal:

```powershell
cd datheon-machine-learning-internship
git init
git add .
git commit -m "Complete Datheon ML internship: classification, churn and NLP sentiment projects"
git branch -M main
git remote add origin https://github.com/salikhussain71-code/datheon-machine-learning-internship.git
git push -u origin main
```

3. On GitHub: **About (gear icon)** → paste the description, add the topics, tick nothing else.
4. Open each notebook on GitHub and confirm the plots are visible. Check the **Actions** tab turns green (the tests run automatically).
5. Optional release: **Releases → Create new release → tag `v1.0.0` → title "Datheon ML Internship — final submission"**.

## If `git push` asks for a password
GitHub no longer accepts account passwords for Git. Use **GitHub Desktop**, the **Git Credential Manager** browser sign-in that Git for Windows opens, or a Personal Access Token (Settings → Developer settings → Tokens) when prompted.

## Check before you submit
* No secrets in the repo (there are none: only public datasets).
* Large generated files: the 7 MB `movie_reviews.csv` cache is already git-ignored; saved models total about 5.5 MB, which is fine for GitHub.

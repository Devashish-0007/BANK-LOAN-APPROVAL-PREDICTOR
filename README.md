# 🏦 Bank Loan Approval Predictor

A small machine-learning project that predicts whether a bank loan application will be **approved or rejected**. It trains a Random Forest on the Kaggle *Loan Prediction Problem Dataset* and ships as a **Google Colab notebook** and an interactive **Streamlit web app**.

> Mini project for the MBA (Data Science and Artificial Intelligence), Chandigarh University, 2026–2027.

## Features

- Downloads the dataset straight from Kaggle with `kagglehub`
- Quick exploration: missing values, class balance, effect of credit history
- One scikit-learn pipeline: imputation, one-hot encoding and Random Forest, so nothing leaks from the test set
- Evaluation with a classification report, confusion matrix and feature importance
- Streamlit app: enter an applicant's details and get an approved/rejected prediction with a confidence value

## Project structure

| File | Purpose |
|---|---|
| `bank_loan_prediction.ipynb` | Colab notebook: download, explore, train, evaluate |
| `app.py` | Streamlit web app |
| `requirements.txt` | Python dependencies |

## Dataset

[Loan Prediction Problem Dataset](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset) on Kaggle: 614 applications, 11 input attributes (gender, marital status, dependents, education, employment, income, co-applicant income, loan amount, loan term, credit history, property area) and the target `Loan_Status` (Y/N).

## Model

| Setting | Value |
|---|---|
| Algorithm | `RandomForestClassifier` (200 trees, max depth 5) |
| Preprocessing | Median imputation (numeric), most-frequent imputation + one-hot encoding (categorical) |
| Split | 80:20, stratified, `random_state=42` |

## Results

| Measure | Random Forest | Always predict "approved" |
|---|---|---|
| Accuracy on the 123-application test set | **85.4%** | 69.1% |
| Mean accuracy, 5-fold cross-validation | **80.8%** | 68.7% |

Credit history is by far the most important feature. The model rarely rejects a good applicant, but it detects only about 55% of the applications that were actually rejected. The cross-validation and baseline figures come from the project report.

## How to run

**Notebook (Google Colab):** open [colab.research.google.com](https://colab.research.google.com), choose *File → Upload notebook*, select `bank_loan_prediction.ipynb`, then *Runtime → Run all*.

**Web app (locally):**

```bash
pip install -r requirements.txt
streamlit run app.py
```

**Web app (in Colab):** upload `app.py`, then run:

```python
!pip install -q streamlit kagglehub
!curl ipv4.icanhazip.com   # this IP is the tunnel password
!streamlit run app.py &>/dev/null & npx localtunnel --port 8501
```

If `kagglehub` asks for credentials, add your Kaggle API key (`kaggle.json`), or download the CSV manually and load it with `pandas.read_csv`.

## Tech stack

Python · pandas · scikit-learn · seaborn / Matplotlib · Streamlit · kagglehub · Google Colab

## Limitations

- Small dataset (614 records), so the estimates are uncertain.
- The model is not tuned for a bank's risk preference and may approve loans that should be rejected.
- The data has no information on existing debts, employment history or collateral, and fairness was not evaluated.
- This is an academic prototype. **Do not use it for real lending decisions.**

## Author

Devashish (25MDA10067)

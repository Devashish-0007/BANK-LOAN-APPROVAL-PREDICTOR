import glob

import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

st.set_page_config(page_title="Loan Approval Predictor", page_icon="🏦")


@st.cache_resource(show_spinner="Downloading data and training model...")
def train_model():
    import kagglehub

    path = kagglehub.dataset_download("altruistdelhite04/loan-prediction-problem-dataset")
    # Only the train file has the Loan_Status label
    df = next(d for d in (pd.read_csv(f) for f in glob.glob(f"{path}/*.csv")) if "Loan_Status" in d)
    df = df.drop(columns="Loan_ID")

    X, y = df.drop(columns="Loan_Status"), df["Loan_Status"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    prep = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), selector(dtype_include="number")),
        ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                              OneHotEncoder(handle_unknown="ignore")),
         selector(dtype_exclude="number")),
    ])
    model = make_pipeline(prep, RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42))
    model.fit(X_train, y_train)
    return model, model.score(X_test, y_test)


model, accuracy = train_model()

st.title("🏦 Bank Loan Approval Predictor")
st.caption(f"Random Forest trained on the Kaggle Loan Prediction dataset · test accuracy {accuracy:.0%}")

with st.form("applicant"):
    c1, c2 = st.columns(2)
    gender = c1.selectbox("Gender", ["Male", "Female"])
    married = c2.selectbox("Married", ["Yes", "No"])
    dependents = c1.selectbox("Dependents", ["0", "1", "2", "3+"])
    education = c2.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = c1.selectbox("Self employed", ["No", "Yes"])
    property_area = c2.selectbox("Property area", ["Urban", "Semiurban", "Rural"])
    income = c1.number_input("Applicant income (monthly)", 0, 100000, 5000, step=500)
    co_income = c2.number_input("Co-applicant income (monthly)", 0, 100000, 0, step=500)
    loan_amount = c1.number_input("Loan amount (in thousands)", 1, 700, 130)
    term = c2.selectbox("Loan term (months)", [360, 180, 480, 300, 240, 120, 84, 60, 36, 12])
    credit = st.radio("Credit history", ["Good (meets guidelines)", "Bad"], horizontal=True)
    submitted = st.form_submit_button("Check eligibility", type="primary")

if submitted:
    applicant = pd.DataFrame([{
        "Gender": gender, "Married": married, "Dependents": dependents,
        "Education": education, "Self_Employed": self_employed,
        "ApplicantIncome": income, "CoapplicantIncome": co_income,
        "LoanAmount": loan_amount, "Loan_Amount_Term": float(term),
        "Credit_History": 1.0 if credit.startswith("Good") else 0.0,
        "Property_Area": property_area,
    }])
    proba = model.predict_proba(applicant)[0][list(model.classes_).index("Y")]
    if proba >= 0.5:
        st.success(f"✅ Likely to be **approved** ({proba:.0%} confidence)")
    else:
        st.error(f"❌ Likely to be **rejected** ({1 - proba:.0%} confidence)")
    st.progress(float(proba), text=f"Approval probability: {proba:.0%}")

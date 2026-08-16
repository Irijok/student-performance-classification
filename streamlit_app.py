import streamlit as st
import pandas as pd
import joblib
import time
import os
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize
# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Student Performance Classification",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# ICONS
# ==========================================

# ==========================================
# UNICODE ICONS
# ==========================================

SCHOOL_ICON = "\U0001F3EB"          
STUDENT_ICON = "\U0001F393"         
UPLOAD_ICON = "\U0001F4C2"          
MODEL_ICON = "\U0001F916"          
ROCKET_ICON = "\U0001F680"         
SUCCESS_ICON = "\u2705"             
TABLE_ICON = "\U0001F4CB"          
BOOK_ICON = "\U0001F4DA"            
STAR_ICON = "\u2B50"                
CHART_ICON = "\U0001F4CA"          
DOWNLOAD_ICON = "\U0001F4E5"       
BACK_ICON = "\u2B05\uFE0F"          
GREEN_ICON = "\U0001F7E2"          
YELLOW_ICON = "\U0001F7E1"          
RED_ICON = "\U0001F534"            
PERSON_ICON = "\U0001F464"          
FILE_ICON = "\U0001F4C4"            

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

.stApp{
background:linear-gradient(135deg,#E3F2FD,#FFFFFF,#F3E5F5);
}

.main-title{
text-align:center;
font-size:42px;
font-weight:bold;
color:#1565C0;
margin-bottom:10px;
}

.subtitle{
text-align:center;
font-size:20px;
color:#444;
margin-bottom:25px;
}

.card{
background:white;
padding:20px;
border-radius:15px;
box-shadow:0px 4px 15px rgba(0,0,0,.15);
transition:.3s;
}

.card:hover{
transform:translateY(-6px);
box-shadow:0px 8px 20px rgba(0,0,0,.25);
}

.stButton>button{
background:#1976D2;
color:white;
font-size:18px;
font-weight:bold;
border-radius:10px;
height:55px;
width:100%;
}

.stButton>button:hover{
background:#0D47A1;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# HEADER
# ==========================================

st.markdown(
f'<div class="main-title">{SCHOOL_ICON} Student Performance Classification Dashboard</div>',
unsafe_allow_html=True
)

st.markdown(
'<div class="subtitle">Predict Student Performance using Machine Learning</div>',
unsafe_allow_html=True
)

# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.success(
    f"Upload a CSV file and choose a model."
)

if "selected_model" not in st.session_state:
    st.session_state["selected_model"] = "Logistic Regression"
models = [
    "Logistic Regression",
    "Decision Tree",
    "KNN",
    "Naive Bayes",
    "Random Forest"
]

model_choice = st.sidebar.selectbox(
    "Select Classification Model",
    models,
    key="model_select"
)

# If the model has changed, clear previous prediction results
if "selected_model" in st.session_state:
    if st.session_state["selected_model"] != model_choice:

        keys_to_remove = [
            "results",
            "metrics",
            "excellent",
            "average",
            "fail",
            "total"
        ]

        for key in keys_to_remove:
            st.session_state.pop(key, None)

st.session_state["selected_model"] = model_choice
st.session_state["prediction_model"] = model_choice
# ==========================================
# MODEL PATHS
# ==========================================

model_paths = {

"Logistic Regression":"model/logistic_regression.pkl",

"Decision Tree":"model/decision_tree.pkl",

"KNN":"model/knn.pkl",

"Naive Bayes":"model/naive_bayes.pkl",

"Random Forest":"model/random_forest.pkl"

}

model = joblib.load(model_paths[model_choice])

scaler = joblib.load("model/scaler.pkl")

feature_columns = joblib.load("model/feature_columns.pkl")
metrics_data = joblib.load("model/metrics.pkl")


# ==========================================
# MAIN PAGE
# ==========================================

st.markdown("---")

uploaded = st.file_uploader(

f"{UPLOAD_ICON} Upload CSV Dataset",

type=["csv"]

)

if uploaded:

    data = pd.read_csv(uploaded)

    st.success("Dataset uploaded successfully.")

    c1,c2,c3 = st.columns(3)

    c1.metric("Rows",len(data))

    c2.metric("Columns",len(data.columns))

    c3.metric("Selected Model",model_choice)

    st.subheader(f"{TABLE_ICON} Dataset Preview")

    st.dataframe(

        data,

        use_container_width=True,

        height=350

    )

generate = st.button(
    f"Generate Predictions",
    disabled=(uploaded is None)
)

if generate:

    # ===============================
    # Prepare Features
    # ===============================
    X = data.copy()

    if "G3" in X.columns:
        X = X.drop(columns=["G3"])

    if "Performance" in X.columns:
        y_true = X["Performance"]
        X = X.drop(columns=["Performance"])
    else:
        y_true = None

    if "Unnamed: 0" in X.columns:
        X = X.drop(columns=["Unnamed: 0"])

    X = pd.get_dummies(
        X,
        drop_first=True
    )

    X = X.reindex(
        columns=feature_columns,
        fill_value=0
    )

    # Scale only LR and KNN
    if model_choice in ["Logistic Regression", "KNN"]:
        X = scaler.transform(X)

    # ===============================
    # Generate Predictions
    # ===============================
    pred = model.predict(X)

    # ===============================
    # Evaluation Metrics
    # ===============================
    metrics = None

    if y_true is not None:

        accuracy = accuracy_score(y_true, pred)

        precision = precision_score(
            y_true,
            pred,
            average="weighted",
            zero_division=0
        )

        recall = recall_score(
            y_true,
            pred,
            average="weighted",
            zero_division=0
        )

        f1 = f1_score(
            y_true,
            pred,
            average="weighted",
            zero_division=0
        )

        mcc = matthews_corrcoef(
            y_true,
            pred
        )

        try:

            if hasattr(model, "predict_proba"):

                probs = model.predict_proba(X)

                y_true_bin = label_binarize(
                    y_true,
                    classes=model.classes_
                )

                auc = roc_auc_score(
                    y_true_bin,
                    probs,
                    average="weighted",
                    multi_class="ovr"
                )

            else:
                auc = None

        except Exception:
            auc = None

        metrics = {
            "Accuracy": accuracy,
            "AUC": auc,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "MCC": mcc
        }

        st.session_state["metrics"] = metrics

    # ===============================
    # Prediction Labels
    # ===============================
    prediction_labels = []

    for p in pred:

        if p == "Excellent":
            prediction_labels.append("Excellent")

        elif p == "Average":
            prediction_labels.append("Average")

        else:
            prediction_labels.append("Fail")

    results = pd.DataFrame({
        "Student ID": range(1, len(pred) + 1),
        "Prediction": prediction_labels
    })

    excellent = sum(pred == "Excellent")
    average = sum(pred == "Average")
    fail = sum(pred == "Fail")

    st.session_state["results"] = results
    st.session_state["excellent"] = excellent
    st.session_state["average"] = average
    st.session_state["fail"] = fail
    st.session_state["total"] = len(pred)
    st.session_state["selected_model"] = model_choice

    st.success("Predictions generated successfully!")
    metrics_data = joblib.load("model/metrics.pkl")

    st.session_state["metrics"] = metrics_data[model_choice]
    st.switch_page("pages/1_Prediction_Results.py")
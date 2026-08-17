"""
train_models.py
================
Trains all 5 required classifiers on the Student Performance dataset
and saves every .pkl file the Streamlit app (streamlit_app.py) expects:

    model/logistic_regression.pkl
    model/decision_tree.pkl
    model/knn.pkl
    model/naive_bayes.pkl
    model/random_forest.pkl
    model/scaler.pkl
    model/feature_columns.pkl
    model/metrics.pkl

Run this ONCE (offline) before running the Streamlit app.

USAGE
-----
Just run it as-is:
    python train_models.py

Or point it at a specific CSV (useful in a virtual lab / different machine):
    python train_models.py --data /path/to/por2_1.csv

Or set an environment variable instead:
    export STUDENT_DATA_PATH=/path/to/por2_1.csv
    python train_models.py
"""

import os
import sys
import argparse
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

RANDOM_STATE = 42
MODEL_DIR = "model"
DEFAULT_FILENAME = "por2_1.csv"


# ==========================================
# 0. RESOLVE DATA PATH (portable across machines)
# ==========================================
def resolve_data_path():
    """
    Figures out where the dataset CSV lives, in this order of priority:
      1. --data command-line argument
      2. STUDENT_DATA_PATH environment variable
      3. A file named 'por2_1.csv' in the current working directory
      4. A file named 'por2_1.csv' in the same directory as this script
    Raises a clear error if none of these resolve to an existing file.
    """
    parser = argparse.ArgumentParser(description="Train student performance models.")
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to the dataset CSV (e.g. por2_1.csv). Overrides everything else.",
    )
    args, _ = parser.parse_known_args()

    candidates = []

    if args.data:
        candidates.append(args.data)

    env_path = os.environ.get("STUDENT_DATA_PATH")
    if env_path:
        candidates.append(env_path)

    candidates.append(os.path.join(os.getcwd(), DEFAULT_FILENAME))

    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates.append(os.path.join(script_dir, DEFAULT_FILENAME))

    for path in candidates:
        if path and os.path.isfile(path):
            print(f"Using dataset: {path}")
            return path

    tried = "\n  - ".join(candidates)
    raise FileNotFoundError(
        "Could not find the dataset CSV. Tried:\n  - " + tried +
        "\n\nFix this by either:\n"
        "  1) Placing 'por2_1.csv' in the same folder as this script, or\n"
        "  2) Running: python train_models.py --data /full/path/to/por2_1.csv, or\n"
        "  3) Setting: export STUDENT_DATA_PATH=/full/path/to/por2_1.csv"
    )


DATA_PATH = resolve_data_path()
os.makedirs(MODEL_DIR, exist_ok=True)

# ==========================================
# 1. LOAD DATA
# ==========================================
df = pd.read_csv(DATA_PATH)

if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

# ==========================================
# 2. CREATE TARGET: "Performance"
#    (built from the final grade G3, Portuguese 0-20 scale)
#    Fail      : G3 < 10
#    Average   : 10 <= G3 < 15
#    Excellent : G3 >= 15
# ==========================================
def label_performance(g3):
    if g3 < 10:
        return "Fail"
    elif g3 < 15:
        return "Average"
    else:
        return "Excellent"

df["Performance"] = df["G3"].apply(label_performance)

print("Class distribution:")
print(df["Performance"].value_counts(), "\n")

# ==========================================
# 3. FEATURE / TARGET SPLIT
#    Mirrors exactly what streamlit_app.py does when it
#    prepares the uploaded CSV for prediction, so training
#    and inference features stay perfectly aligned.
# ==========================================
y = df["Performance"]

X = df.drop(columns=["Performance", "G3"])  # G3 dropped: it's what the target is derived from (leakage)

# One-hot encode categoricals exactly like the app does
X = pd.get_dummies(X, drop_first=True)

# Save the exact column order/set used for training -> app reindexes new data to this
feature_columns = X.columns.tolist()
joblib.dump(feature_columns, os.path.join(MODEL_DIR, "feature_columns.pkl"))

# ==========================================
# 4. TRAIN / TEST SPLIT
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y,
)

# ==========================================
# 5. SCALING (only Logistic Regression & KNN use scaled data,
#    matching the app's: if model_choice in ["Logistic Regression", "KNN"])
# ==========================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

# ==========================================
# 6. DEFINE MODELS
# ==========================================
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
}

# Map to the exact filenames streamlit_app.py's model_paths dict expects
model_filenames = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "KNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest": "random_forest.pkl",
}

classes = sorted(y.unique())  # for multiclass AUC binarization
y_test_bin = label_binarize(y_test, classes=classes)

# Save class label order -> the app uses this to label the confusion matrix axes
joblib.dump(classes, os.path.join(MODEL_DIR, "classes.pkl"))

metrics_data = {}

# ==========================================
# 7. TRAIN, EVALUATE, SAVE EACH MODEL
# ==========================================
for name, model in models.items():

    uses_scaled = name in ["Logistic Regression", "KNN"]
    X_tr = X_train_scaled if uses_scaled else X_train
    X_te = X_test_scaled if uses_scaled else X_test

    model.fit(X_tr, y_train)
    y_pred = model.predict(X_te)

    # Multiclass AUC (One-vs-Rest, weighted) via predict_prob
    try:
        y_proba = model.predict_proba(X_te)
        auc = roc_auc_score(y_test_bin, y_proba, multi_class="ovr", average="weighted")
    except Exception:
        auc = None

    # Confusion matrix, ordered consistently using the saved `classes` list
    cm = confusion_matrix(y_test, y_pred, labels=classes)

    # Classification report as a dict (per-class precision/recall/f1/support)
    # so the app can render it as a table instead of a plain text blob.
    report_dict = classification_report(
        y_test, y_pred, labels=classes, output_dict=True, zero_division=0
    )

    metrics_data[name] = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "F1 Score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "MCC": matthews_corrcoef(y_test, y_pred),
        "AUC": auc,
        "ConfusionMatrix": cm,              # numpy array, shape (n_classes, n_classes)
        "ClassificationReport": report_dict,  # dict, e.g. report_dict["Excellent"]["f1-score"]
    }

    joblib.dump(model, os.path.join(MODEL_DIR, model_filenames[name]))

    if auc is not None:
        print(f"{name:20s} -> Acc: {metrics_data[name]['Accuracy']:.4f} | "
              f"F1: {metrics_data[name]['F1 Score']:.4f} | "
              f"AUC: {auc:.4f}")
    else:
        print(f"{name}: AUC N/A")

# ==========================================
# 8. SAVE METRICS.PKL  (keyed by model name, matches app lookup)
# ==========================================
joblib.dump(metrics_data, os.path.join(MODEL_DIR, "metrics.pkl"))

print("\nAll .pkl files saved to:", os.path.abspath(MODEL_DIR))
print(os.listdir(MODEL_DIR))

# Student-performance-classification
A comprehensive Student Performance Classification app implementing five ML models—Logistic Regression, Decision Tree, KNN, Naive Bayes, and Random Forest—with Accuracy, AUC, Precision, Recall, F1, and MCC metrics through an interactive Streamlit dashboard.
# Student Performance Classification

## a. Problem Statement

Student academic performance can be influenced by several factors such as study habits, previous failures, family background, social activities, attendance, and other personal and academic characteristics. Identifying student performance categories using these factors can help in understanding academic outcomes and supporting data-driven educational decisions.

The objective of this project is to develop a **Student Performance Classification application** using multiple machine learning classification algorithms. The application predicts student performance categories and provides a detailed comparison of the implemented models using standard classification evaluation metrics.

An interactive **Streamlit dashboard** has also been developed to allow users to upload student data, select a classification model, generate predictions, and view the corresponding performance results and evaluation metrics.

---

## b. Dataset Description

The dataset used for this project is the **Student Performance dataset** (`por2.csv`), containing information related to students' academic, demographic, family, social, and lifestyle characteristics.

### Dataset Statistics

| Property            | Details             |
| ------------------- | ------------------- |
| Number of Instances | 649                 |
| Number of Columns   | 35 / 34           |
| Target              | Student Performance |
| Dataset Type        | Classification      |
| File                | `por2_1.csv` / `por2.csv'     |

### Important Features

The dataset contains several types of attributes, including:

* **Demographic information:** age, sex, address
* **Family information:** family size, parents' education, parents' jobs, guardian
* **Academic factors:** study time, previous failures, school support
* **Social factors:** going out, free time, romantic relationship
* **Lifestyle factors:** alcohol consumption, health, absences
* **Academic grades:** G1, G2, and G3

The student performance information is used to classify students into performance categories such as **Excellent, Average, and Fail**.

---

## c. GitHub Repository Link

**GitHub Repository:**
https://github.com/Irijok/student-performance-classification

This repository contains the complete implementation of the Student Performance Classification project, including the Streamlit application, trained machine learning models, supporting files, test data, and project documentation.

---

## d. Models Used

Five machine learning classification models were implemented on the same dataset:

1. **Logistic Regression**
2. **Decision Tree Classifier**
3. **K-Nearest Neighbor (KNN) Classifier**
4. **Naive Bayes Classifier**
5. **Random Forest Classifier (Ensemble Model)**

Each model was evaluated using the following six metrics:

* Accuracy
* AUC Score
* Precision
* Recall
* F1 Score
* Matthews Correlation Coefficient (MCC)

### Model Comparison

| ML Model Name            | Accuracy | AUC   | Precision | Recall | F1     | MCC    |
| ------------------------ | -------: | --:   | --------: | -----: | -:     | --:    |
| Logistic Regression      |   0.8692 | 0.9471|    0.8696 | 0.8692 | 0.8691 | 0.7497 |
| Decision Tree            |   0.8615 | 0.8787|    0.8683 | 0.8615 | 0.8630 | 0.7464 |
| KNN                      |   0.7154 | 0.7091|    0.7403 | 0.7154 | 0.6861 | 0.3538 |
| Naive Bayes              |   0.6231 | 0.8853|    0.7650 | 0.6231 | 0.6260 | 0.4921 |
| Random Forest (Ensemble) |   0.8846 | 0.9694|    0.8865 | 0.8846 | 0.8752 | 0.7731 |


Overall Winner: **Random Forest (Ensemble)**


ML Model	Observation
Logistic Regression	- Strong overall performance with 86.92% accuracy and high AUC (0.9471), making it a reliable baseline model.
Decision Tree -	Provides good classification performance with 86.15% accuracy, but performs slightly below Logistic Regression and Random Forest.
KNN	- Shows moderate performance with 71.54% accuracy and lower MCC/F1 scores, indicating weaker overall classification ability.
Naive Bayes -	Has the lowest accuracy (62.31%) and relatively low F1 and MCC, making it the weakest model overall despite a reasonably high AUC.
Random Forest (Ensemble) -	Best overall model with the highest Accuracy (88.46%), AUC (0.9694), Precision (0.8846), and MCC (0.7731).


Overall Winner


**Random Forest (Ensemble)** is the overall best-performing model for the Student Performance Classification dataset. It achieves the highest Accuracy, AUC, Precision, and MCC while maintaining a strong Recall and F1 Score. Therefore, Random Forest can be selected as the preferred model for this application.


## Application Features

The project includes an interactive **Streamlit Student Performance Classification Dashboard**.

The application provides:

* CSV dataset upload
* Dataset preview
* Selection of classification model
* Student performance predictions
* Prediction distribution
* Model evaluation metrics
* Accuracy, AUC, Precision, Recall, F1 Score, and MCC
* Interactive prediction results
* Downloadable prediction results
* User-friendly dashboard interface

---

## Machine Learning Workflow

The overall workflow of the project is:

```text
Dataset
   ↓
Data Preprocessing
   ↓
Feature Encoding
   ↓
Feature Selection
   ↓
Train/Test Split
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Model Saving
   ↓
Streamlit Application
   ↓
Student Performance Prediction
```

---

## Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **Joblib**
* **Streamlit**
---
## Repository Structure
```
student-performance-classification/
│
├── streamlit_app.py
├── requirements.txt
├── README.md
├── test_data.csv
│
├── model/
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── knn.pkl
│   ├── naive_bayes.pkl
│   ├── random_forest.pkl
│   ├── scaler.pkl
│   ├── feature_columns.pkl
│   └── metrics.pkl
│
└── pages/
|    └── 1_Prediction_Results.py
── train/
|    └── train_models.py
├── por2_1.csv
├── por2.csv
```

---

## How to Run the Application

### 1. Clone the repository

```bash
git clone <https://github.com/Irijok/student-performance-classification>
```

### 2. Navigate to the project directory

```bash
cd student-performance-classification
```

### 3. Install the required packages

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit application

```bash
streamlit run streamlit_app.py
```

The application will open in the browser.

---

## Basics

To generate pkl files, run the train_models.py in train folder.
Then after making sure that all the files exist within the same and accessible path, run streamlit_app.py. If needed, change the DATA_PATH to the particular path in which the csv file is downloaded within the system.

## Conclusion

This project demonstrates the application of multiple machine learning classification algorithms to student performance data. The implemented models are evaluated using five different performance metrics, allowing their effectiveness to be compared systematically.

The Streamlit application provides an interactive interface for uploading data, selecting machine learning models, generating student performance predictions, and viewing detailed evaluation results.

The project demonstrates the complete machine learning workflow, from data preprocessing and model development to evaluation, model saving, and deployment through an interactive web application.


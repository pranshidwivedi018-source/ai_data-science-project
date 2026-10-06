# 🚢 Titanic Survival Prediction — Machine Learning Project

A complete Machine Learning project that predicts whether a passenger survived the Titanic disaster using passenger information such as class, sex, age, family relationships, fare, and port of embarkation.

The project demonstrates a practical ML workflow: **data preprocessing → feature engineering → feature selection → model training → prediction → evaluation → model saving**.

---

## 👥 Team Members

| Name | Roll No. | Enrollment No. | Role |
|---|---:|---|---|
| **Pranshi Dwivedi** | 25225100021 | CSJMA25000006150 | **Team Captain** |
| Anshika Shukla | 2522510006 | CSJMA25000035668 | Team Member |
| Rishabh Tiwari | 255225100025 | CSJMA25000006152 | Team Member |
| Bhanu Pratap Singh Katiyar | 25225100009 | CSJMA252500089656 | Team Member |

---

## 📌 Project Overview

The Titanic Survival Prediction project uses historical passenger data from the Titanic to build a binary classification model.

The model learns patterns from passenger information and predicts one of two outcomes:

- **0 → Not Survived**
- **1 → Survived**

The final model is **Logistic Regression**, a commonly used classification algorithm for predicting binary outcomes.

---

## 🎯 Problem Statement

The Titanic disaster resulted in many passenger deaths. Survival was influenced by factors such as passenger class, sex, age, fare, family relationships, and embarkation port.

**Problem:**  
Build a Machine Learning model that can predict whether a passenger would survive based on the available passenger information.

---

## 🧠 What Does the Model Do?

The model takes passenger-related features as input, processes them using the same preprocessing steps used during training, and predicts:

```text
Passenger Information
        ↓
Data Preprocessing
        ↓
Feature Engineering
        ↓
Feature Selection
        ↓
Logistic Regression
        ↓
Survival Prediction
        ↓
0 = Not Survived / 1 = Survived
```

---

## 📊 Dataset

The project uses the standard Titanic training dataset (`train.csv`).

### Original Dataset

- **Rows:** 891
- **Columns:** 12
- **Target:** `Survived`

### Original Columns

```text
PassengerId
Survived
Pclass
Name
Sex
Age
SibSp
Parch
Ticket
Fare
Cabin
Embarked
```

### Target Variable

| Value | Meaning |
|---:|---|
| 0 | Not Survived |
| 1 | Survived |

---

## 🔧 Data Preprocessing

The dataset contains missing values and categorical variables, so preprocessing is performed before model training.

### 1. Missing Values

**Age**

Missing age values are filled using the **median calculated from the training data**.

**Embarked**

Missing values are filled using the **mode calculated from the training data**.

**Cabin**

The `Cabin` column contains a large number of missing values, so it is removed from the final modeling dataset.

### 2. Removed Columns

The following columns are removed because they are not directly used as final predictive features:

```text
PassengerId
Name
Ticket
Cabin
```

---

## ⚙️ Feature Engineering

Several transformations are applied to prepare the data for Machine Learning.

### Sex Encoding

The categorical `Sex` feature is converted into a numerical representation.

### One-Hot Encoding

The `Embarked` feature is converted into:

```text
Embarked_C
Embarked_Q
Embarked_S
```

### Fare Transformation

A logarithmic transformation is applied to `Fare`:

```text
Fare_log
```

This helps reduce the effect of highly skewed fare values.

### Standardization

Numerical features are standardized using **Z-score standardization**.

The transformed features include:

```text
Pclass_std
Age_std
SibSp_std
Parch_std
Fare_log_std
```

---

## 🔍 Feature Selection

The notebook explores multiple statistical and ML-based feature-selection techniques:

- Variance Threshold
- Pearson Correlation
- Chi-Square Test
- ANOVA F-Test
- Mutual Information

These techniques help understand which features contain useful information for predicting survival.

---

## ⭐ Final Features Used by the Model

The final Logistic Regression model uses:

```text
Pclass_std
Sex
Age_std
SibSp_std
Parch_std
Fare_log_std
Embarked_C
Embarked_Q
Embarked_S
```

---

## 🤖 Machine Learning Model

### Logistic Regression

The final model is:

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)
```

Logistic Regression is suitable for this project because the target variable has two possible classes:

```text
0 → Not Survived
1 → Survived
```

The model estimates the probability of survival and converts it into a class prediction.

---

## 📚 Train-Test Split

The dataset is divided into:

- **80% training data**
- **20% testing data**

The test set contains **179 passengers**.

The model learns patterns from the training data and is evaluated on the unseen test data.

---

## 🔮 Prediction

Predictions are generated using:

```python
y_pred = model.predict(X_test)
```

The predictions are compared with the actual survival values to measure model performance.

---

## 📈 Model Performance

### Test Accuracy

The final Logistic Regression model achieved:

# **83.24% Accuracy**

This means the model correctly classified approximately 83 out of every 100 passengers in the **179-passenger held-out test set**.

> **Important:** 83.24% is the accuracy on this project's test set, not a guarantee of performance on all Titanic passengers or future datasets.

---

## 📊 Confusion Matrix

The model produced the following confusion matrix:

```text
[[103, 12],
 [ 18, 46]]
```

| | Predicted Not Survived | Predicted Survived |
|---|---:|---:|
| **Actual Not Survived** | 103 | 12 |
| **Actual Survived** | 18 | 46 |

### Interpretation

- **True Negative (TN): 103** — correctly predicted passengers who did not survive.
- **False Positive (FP): 12** — predicted survival when the passenger did not survive.
- **False Negative (FN): 18** — predicted non-survival when the passenger actually survived.
- **True Positive (TP): 46** — correctly predicted passengers who survived.

---

## 📋 Classification Report

| Class | Precision | Recall | F1-Score | Support |
|---|---:|---:|---:|---:|
| Not Survived | 0.85 | 0.90 | 0.87 | 115 |
| Survived | 0.79 | 0.72 | 0.75 | 64 |
| **Accuracy** | | | **0.83** | **179** |
| Macro Avg | 0.82 | 0.81 | 0.81 | 179 |
| Weighted Avg | 0.83 | 0.83 | 0.83 | 179 |

---

## 💾 Model Saving

After training, the Logistic Regression model is saved using Python's `pickle` module:

```python
with open('logistic_regression_model.pkl', 'wb') as f:
    pickle.dump(model, f)
```

This allows the trained model to be reused without training it again.

---

## 🛠️ Technologies Used

- **Python**
- **Pandas**
- **NumPy**
- **Matplotlib**
- **SciPy**
- **Scikit-learn**
- **Jupyter Notebook**
- **Google Colab**
- **GitHub**

---

## 📁 Project Structure

```text
Titanic-Survival-Prediction/
│
├── Titanic_survial_Prediction (1).ipynb
├── train.csv
├── logistic_regression_model.pkl
└── README.md
```

> File names may vary depending on how the project is uploaded to GitHub.

---

## ▶️ How to Run the Project

### Option 1 — Google Colab

1. Open the notebook.
2. Upload `train.csv` when requested.
3. Run the notebook cells from top to bottom.
4. The notebook performs preprocessing, feature engineering, feature selection, model training, and evaluation.
5. View the final accuracy and evaluation results.

### Option 2 — Jupyter Notebook

Install the required libraries:

```bash
pip install pandas numpy matplotlib scipy scikit-learn
```

Then open:

```text
Titanic_survial_Prediction (1).ipynb
```

Make sure `train.csv` is available in the expected location and run the notebook.

---

## 🌍 Real-World Machine Learning Relevance

Although this project uses the Titanic dataset, the workflow represents a common real-world Machine Learning process.

Similar classification approaches can be used for:

- Customer churn prediction
- Loan approval prediction
- Disease-risk classification
- Fraud detection
- Employee attrition prediction
- Customer response prediction

The important learning is not only the final accuracy, but the complete process of converting raw data into a usable predictive model.

---

## 🚀 Future Improvements

The project can be improved by:

- Comparing Logistic Regression with Decision Tree, Random Forest, SVM, and other classifiers.
- Applying hyperparameter tuning.
- Performing cross-validation.
- Handling class imbalance if required.
- Creating a user-friendly prediction interface.
- Deploying the model as a web application.
- Adding more detailed exploratory data analysis.
- Comparing different feature-selection strategies using consistent validation.

---

## 👨‍💻 Team Contributions

### Pranshi Dwivedi — Team Captain
- Overall project coordination
- Machine Learning workflow
- Model integration
- Final project organization

### Anshika Shukla
- Dataset understanding
- Data preprocessing
- Exploratory data analysis

### Rishabh Tiwari
- Feature engineering
- Feature selection
- Statistical analysis

### Bhanu Pratap Singh Katiyar
- Model evaluation
- Testing
- Documentation and presentation

> Contributions can be adjusted to reflect the team's actual individual work.

---

## 🎓 Learning Outcomes

Through this project, the team learned how to:

- Understand a real-world dataset.
- Identify and handle missing values.
- Encode categorical data.
- Transform and standardize numerical features.
- Perform feature selection.
- Train a classification model.
- Generate predictions on unseen data.
- Evaluate a model using accuracy, precision, recall, F1-score, and confusion matrix.
- Save a trained Machine Learning model.
- Document an ML project for GitHub.

---

## ✅ Conclusion

This project demonstrates a complete Machine Learning pipeline for **Titanic Survival Prediction**.

After preprocessing the Titanic dataset, engineering useful features, exploring feature-selection techniques, and training a **Logistic Regression** classifier, the model achieved **83.24% test accuracy** on 179 unseen test samples.

The project provides a practical foundation for understanding how Machine Learning can transform historical data into predictive insights.

---

## ⭐ Project Highlights

```text
Dataset        → Titanic train.csv
Problem        → Binary Classification
Target         → Survived
Model          → Logistic Regression
Train/Test     → 80% / 20%
Test Samples   → 179
Accuracy       → 83.24%
Evaluation     → Accuracy, Confusion Matrix, Classification Report
Model Saving   → Pickle (.pkl)
```

---

### 🚢 From Raw Passenger Data to Machine Learning Prediction

**Data → Preprocessing → Features → Model → Prediction → Evaluation**

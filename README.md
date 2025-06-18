# 💳 Credit Scoring Model – CodeAlpha Internship Project

This project aims to predict whether a loan applicant is creditworthy based on historical financial data using machine learning models.

---

## 📁 Dataset

- **Name**: German Credit Data
- **Source**: [UCI Statlog (German Credit Data)](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data))
- **Records**: 1000 applicants
- **Features**: 24 numerical features (processed version) + 1 target variable

The target column:
- `1` = Good credit risk (creditworthy)
- `2` = Bad credit risk (not creditworthy)

---

## 🧠 Problem Statement

To build a predictive model that classifies individuals as **creditworthy** or **not creditworthy** using classification algorithms and financial attributes such as:
- Credit history
- Loan amount
- Duration
- Age
- Job type
- Number of existing credits

---

## 🛠️ Tools & Libraries

- Python (pandas, numpy, matplotlib, seaborn)
- scikit-learn (Logistic Regression, Random Forest, GridSearchCV)

---

## ⚙️ ML Pipeline

### 1. Data Preprocessing
- Loaded `german.data-numeric` file
- Renamed 24 attributes and one target column
- Standardized features using `StandardScaler`
- Mapped target values to binary (1 → 1, 2 → 0)

### 2. Models Trained
- ✅ Logistic Regression  
- ✅ Random Forest Classifier  
- ✅ Random Forest with GridSearchCV for Hyperparameter Tuning

### 3. Evaluation Metrics
- Accuracy, Precision, Recall, F1-Score
- ROC AUC Score
- Visualizations (heatmaps, class distribution, ROC curves)

---

## 📊 Visualizations

- 📌 **Feature Correlation Heatmap**
- 📌 **Target Class Distribution**
- 📌 **ROC Curve Comparison**

---

## 📈 Model Performance

| Model                | Accuracy | F1-Score | ROC AUC |
|---------------------|----------|----------|---------|
| Logistic Regression | 78%      | 0.77     | 0.80    |
| Random Forest        | 79%      | 0.77     | 0.81    |
| RF (Tuned)           | ~82%     | 0.80+    | 0.83+   |

---

## 🔍 Hyperparameter Tuning

- Used `GridSearchCV` to optimize:
  - `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `bootstrap`
- Improved ROC AUC and F1-score using tuned parameters

---

## 📂 Project Structure

CodeAlpha_CreditScoringModel/
│
├── german.data-numeric
├── credit_model.ipynb # Main notebook
├── credit_model.py # (Optional script version)
└── README.md

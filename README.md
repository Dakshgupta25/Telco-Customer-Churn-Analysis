# Telco-Customer-Churn-Analysis
End-to-end machine learning project to predict telecom customer churn. Includes data cleaning, exploratory data analysis, feature engineering, Logistic Regression and Random Forest models, evaluation using ROC-AUC, and feature importance analysis to derive actionable business insights.

## 📌 Project Overview
Customer churn is a major problem for telecom companies, as acquiring new customers is significantly more expensive than retaining existing ones.  
This project aims to **predict customer churn** using machine learning techniques and identify the key factors that influence customer attrition.

---

## 🎯 Objective
- Predict whether a customer will churn (`Yes` / `No`)
- Understand the most important factors contributing to churn
- Provide actionable business insights for customer retention

---

## 📂 Dataset
- **Source:** IBM Telco Customer Churn Dataset (Kaggle)
- **Records:** ~7,000 customers
- **Features:** Demographics, services used, contract details, billing information
- **Target Variable:** `Churn`

---

## 🔍 Project Workflow

### 1. Business Understanding
- Defined churn prediction as a **binary classification problem**
- Focused on **recall and ROC-AUC** due to class imbalance

### 2. Data Understanding
- Explored dataset structure, data types, and statistics
- Identified incorrect data types and hidden missing values

### 3. Data Cleaning
- Converted `TotalCharges` to numeric
- Handled missing values using median imputation
- Removed non-informative identifier column (`customerID`)

### 4. Exploratory Data Analysis (EDA)
- Analyzed churn distribution
- Studied relationships between churn and:
  - Contract type
  - Tenure
  - Monthly charges
- Visualized correlations between numerical features

### 5. Feature Engineering
- Encoded target variable (`Churn`)
- Binary encoded Yes/No features
- Applied one-hot encoding to categorical variables

### 6. Train-Test Split & Scaling
- 80/20 train-test split with stratification
- Applied feature scaling for Logistic Regression

### 7. Model Building
- **Logistic Regression** (baseline, interpretable model)
- **Random Forest Classifier** (non-linear, ensemble model)

### 8. Model Evaluation
- Classification report
- Confusion matrix
- ROC-AUC score

### 9. Model Interpretation
- Logistic Regression coefficients for feature impact
- Random Forest feature importance analysis

---

## 📊 Results
- Random Forest performed better in capturing complex relationships
- Contract type, tenure, and monthly charges were strong churn indicators

---

## 💡 Business Insights
- Month-to-month customers are more likely to churn
- Customers with high monthly charges have higher churn risk
- Long-tenure customers are less likely to leave

**Recommendation:**  
Target high-risk customers early with retention offers and promote long-term contracts.

---

## 🛠️ Technologies Used
- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- Google Colab

---

## 📁 Project Structure

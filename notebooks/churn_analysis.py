# =========================================
# Import Required Libraries
# =========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# =========================================
# Load Dataset
# =========================================

df = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')

pd.set_option('display.max_columns', None)

df.head()
df.shape
df.info()


# =========================================
# Data Cleaning
# =========================================

# Check missing values
df.isnull().sum()

# Convert TotalCharges to numeric
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Fill missing values using median
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

# Drop identifier column
df = df.drop('customerID', axis=1)


# =========================================
# Feature Engineering
# =========================================

# Encode target variable
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Binary encoding
binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
for col in binary_cols:
    df[col] = df[col].map({'Yes': 1, 'No': 0})

# One-hot encoding
df = pd.get_dummies(df, drop_first=True)


# =========================================
# Exploratory Data Analysis (EDA)
# =========================================

# Churn distribution
sns.countplot(x='Churn', data=df)
plt.title("Churn Distribution")
plt.show()

df['Churn'].value_counts(normalize=True)

# Tenure distribution
sns.histplot(df['tenure'], bins=30, kde=True)
plt.title("Tenure Distribution")
plt.show()

# Monthly charges distribution
sns.histplot(df['MonthlyCharges'], bins=30, kde=True)
plt.title("Monthly Charges Distribution")
plt.show()

# Contract vs Churn (using raw data)
df_raw = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')

sns.countplot(x='Contract', hue='Churn', data=df_raw)
plt.title("Contract Type vs Churn")
plt.show()

# Tenure vs Churn
sns.boxplot(x='Churn', y='tenure', data=df)
plt.title("Tenure vs Churn")
plt.show()

# Monthly Charges vs Churn
sns.boxplot(x='Churn', y='MonthlyCharges', data=df)
plt.title("Monthly Charges vs Churn")
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df[['tenure', 'MonthlyCharges', 'TotalCharges', 'Churn']].corr(), annot=True)
plt.title("Correlation Heatmap")
plt.show()


# =========================================
# Train-Test Split
# =========================================

X = df.drop('Churn', axis=1)
y = df['Churn']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# =========================================
# Feature Scaling
# =========================================

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# =========================================
# Model Building
# =========================================

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_scaled, y_train)

# Random Forest
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)


# =========================================
# Model Evaluation
# =========================================

from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

# Logistic Regression predictions
y_pred_lr = lr.predict(X_test_scaled)
y_prob_lr = lr.predict_proba(X_test_scaled)[:, 1]

# Random Forest predictions
y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:, 1]

# Logistic Regression metrics
print("Logistic Regression Results")
print(classification_report(y_test, y_pred_lr))
print("ROC-AUC:", roc_auc_score(y_test, y_prob_lr))

sns.heatmap(confusion_matrix(y_test, y_pred_lr), annot=True, fmt='d')
plt.title("Logistic Regression Confusion Matrix")
plt.show()

# Random Forest confusion matrix
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d')
plt.title("Random Forest Confusion Matrix")
plt.show()


# =========================================
# Feature Importance (Interpretation)
# =========================================

# Logistic Regression feature importance
importance_lr = pd.Series(lr.coef_[0], index=X.columns).sort_values(ascending=False)
top10_lr = importance_lr.head(10)

plt.figure(figsize=(8, 4))
sns.barplot(x=top10_lr.values, y=top10_lr.index)
plt.title("Top 10 Feature Importance (Logistic Regression)")
plt.xlabel("Coefficient Value")
plt.ylabel("Feature")
plt.show()

# Random Forest feature importance
importance_rf = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
top10_rf = importance_rf.head(10)

plt.figure(figsize=(8, 4))
sns.barplot(x=top10_rf.values, y=top10_rf.index)
plt.title("Top 10 Feature Importance (Random Forest)")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.show()

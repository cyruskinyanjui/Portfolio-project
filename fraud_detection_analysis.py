"""
MOBILE MONEY FRAUD DETECTION ANALYSIS
Dataset: Synthetic Financial Datasets For Fraud Detection (PaySim)
Author: Cyrus Macharia Kinyanjui
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# 1. DATA OVERVIEW
Dataset = pd.read_csv(r"C:\Users\HomePC\Downloads\archive (3)\PS_20174392719_1491204439457_log.csv")
print(Dataset.shape)
print(Dataset.info())
print(Dataset.head())
print(Dataset.isnull().sum())


# 2. CLASS BALANCE & TRANSACTION TYPES
print(Dataset['isFraud'].value_counts())
print(Dataset['isFraud'].value_counts(normalize=True) * 100)
print(Dataset['type'].value_counts())
print(pd.crosstab(Dataset['type'], Dataset['isFraud']))


# 3. BALANCE CONSISTENCY CHECK
Dataset['balanceDiffOrig'] = Dataset['oldbalanceOrg'] - Dataset['amount'] - Dataset['newbalanceOrig']
Dataset['balanceDiffDest'] = Dataset['oldbalanceDest'] + Dataset['amount'] - Dataset['newbalanceDest']

print(Dataset['balanceDiffOrig'].describe())
print(Dataset['balanceDiffDest'].describe())
print(Dataset[(Dataset['oldbalanceDest'] == 0) & (Dataset['newbalanceDest'] == 0)]['isFraud'].value_counts())


# 4. AMOUNT DISTRIBUTION: FRAUD VS. NON-FRAUD
subset = Dataset[Dataset['type'].isin(['CASH_OUT', 'TRANSFER'])]

plt.figure(figsize=(10, 5))
sns.boxplot(x='isFraud', y='amount', data=subset, showfliers=False)
plt.title('Transaction Amount: Fraud vs Non-Fraud (CASH_OUT & TRANSFER)')
plt.xlabel('isFraud')
plt.ylabel('Amount') 
plt.show()

print(subset.groupby('isFraud')['amount'].describe())


# 5. T-TEST — IS THE AMOUNT DIFFERENCE REAL?
fraud_amounts = subset[subset['isFraud'] == 1]['amount']
nonfraud_amounts = subset[subset['isFraud'] == 0]['amount']

t_stat, p_value = stats.ttest_ind(fraud_amounts, nonfraud_amounts, equal_var=False)
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_value}")


# 6. ANOVA — DOES AMOUNT VARY BY TRANSACTION TYPE?
groups = [Dataset[Dataset['type'] == t]['amount'] for t in Dataset['type'].unique()]

f_stat, p_value_anova = stats.f_oneway(*groups)
print(f"F-statistic: {f_stat:.4f}")
print(f"P-value: {p_value_anova}")

print(Dataset.groupby('type')['amount'].mean().sort_values(ascending=False))


# 7. LOGISTIC REGRESSION — FIRST PASS
model_dataframe = Dataset[Dataset['type'].isin(['CASH_OUT', 'TRANSFER'])].copy()
model_dataframe['type_TRANSFER'] = (model_dataframe['type'] == 'TRANSFER').astype(int)

features = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'type_TRANSFER']
X = model_dataframe[features]
y = model_dataframe['isFraud']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

log_reg = LogisticRegression(max_iter=1000, class_weight='balanced')
log_reg.fit(X_train, y_train)

y_pred = log_reg.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Standardize features to fairly compare coefficient magnitudes
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

log_reg_scaled = LogisticRegression(max_iter=1000, class_weight='balanced')
log_reg_scaled.fit(X_train_scaled, y_train)

coef_scaled_df = pd.DataFrame({'feature': features, 'coefficient': log_reg_scaled.coef_[0]})
print(coef_scaled_df.sort_values('coefficient', ascending=False))

# Check for multicollinearity
print(X_train[['oldbalanceDest', 'newbalanceDest', 'oldbalanceOrg', 'newbalanceOrig']].corr())


# 8. LOGISTIC REGRESSION — FINAL MODEL
features_v2 = ['amount', 'oldbalanceOrg', 'oldbalanceDest', 'type_TRANSFER']
X2 = model_dataframe[features_v2]
y2 = model_dataframe['isFraud']

X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2, y2, test_size=0.3, random_state=42, stratify=y2
)

scaler2 = StandardScaler()
X2_train_scaled = scaler2.fit_transform(X2_train)
X2_test_scaled = scaler2.transform(X2_test)

log_reg2 = LogisticRegression(max_iter=1000, class_weight='balanced')
log_reg2.fit(X2_train_scaled, y2_train)

# predict on the SCALED test set, since the model was trained on scaled data
y2_pred = log_reg2.predict(X2_test_scaled)
print(classification_report(y2_test, y2_pred))
print(confusion_matrix(y2_test, y2_pred))

coef2_df = pd.DataFrame({'feature': features_v2, 'coefficient': log_reg2.coef_[0]})
print(coef2_df.sort_values('coefficient', ascending=False))
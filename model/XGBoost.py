import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score, matthews_corrcoef, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Load as a dictionary (default)
dataset = load_breast_cancer()
# Consider

X, y = dataset.data, dataset.target

print(X.shape)
print(y.shape)

(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size= 0.25, stratify = y)

print('Training Features :', X_train.shape)
print('Training Labels :', y_train.shape)
print('Testing Features :', X_test.shape)
print('Testing Labels :', y_test.shape)

# 4. Initialize and train XGBoost
# use_label_encoder=False avoids deprecation warnings in newer versions
xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=4,
    random_state=42
)
xgb_model.fit(X_train, y_train)

# Predict the values for testing set

y_predict = xgb_model.predict(X_test)
y_probs = xgb_model.predict_proba(X_test)[:, 1]

print(y_predict,y_probs)

# Accuracy of the classifier

acc = accuracy_score(y_test, y_predict)
print(acc * 100 , "%")

conf_matrix = confusion_matrix(y_test, y_predict)
print('Confusion Matrix :')
print(conf_matrix)

auc = roc_auc_score(y_test, y_probs)
mcc = matthews_corrcoef(y_test, y_predict)

print(f"AUC-ROC Score: {auc:.4f}")
print(f"MCC Score:     {mcc:.4f}")

# Classification Report - precision, recall

print("\nFull Report:")
print('Classification Report: ')
print(classification_report(y_test, y_predict))
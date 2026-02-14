import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, matthews_corrcoef, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# 1. Load data and define 'features' correctly
data = load_breast_cancer()
features = pd.DataFrame(data.data, columns=data.feature_names)
target = data.target

# 2. View the first 7 rows as you intended
print(features.head(7))

# 3. Split the data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# 4. Initialize and train the Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict the values for testing set

y_predict = rf_model.predict(X_test)
y_probs = rf_model.predict_proba(X_test)[:, 1]

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
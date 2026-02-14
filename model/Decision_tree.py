import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, matthews_corrcoef, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Load as a dictionary (default)
dataset = load_breast_cancer()
# Consider

X = dataset["data"][:,2:]

# consider only the class 2.

y = (dataset["target"]==1).astype(np.bool)

print(X.shape)
print(y.shape)

(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size= 0.25, stratify = y)

print('Training Features :', X_train.shape)
print('Training Labels :', y_train.shape)
print('Testing Features :', X_test.shape)
print('Testing Labels :', y_test.shape)

# Train the classifier using Entrooy

tree = DecisionTreeClassifier(criterion = 'entropy')

tree.fit(X_train, y_train) # for training

# Predict the values for testing set

y_predict = tree.predict(X_test)
y_probs = tree.predict_proba(X_test)[:, 1]

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
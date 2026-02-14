import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, matthews_corrcoef, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Load as a dictionary (default)
data = load_breast_cancer()

# Consider

X = data["data"][:,2:]

# consider only the class 2.
# similar to one-vs-all case.

y = (data["target"]==1).astype(np.bool)

# Split data into train and test

(X_train, X_test, y_train, y_test) = train_test_split(X, y, stratify=y, test_size= 0.3)

# testing size = 30 %
# rest 70 % is used for training
# stratify parameter ensures that observations from each class is are given equal weightage
print("Train and test dataset shape")
print(X_train.shape)
print(X_test.shape)
print(np.unique(y_train))

# build the model

logistic_model = LogisticRegression(solver='lbfgs') # default classfier

logistic_model.fit(X_train,y_train)

# predict values for the test data
y_prob  = logistic_model.predict(X_test)

# y_probs: probabilities for the positive class (used for AUC)
y_probs = logistic_model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_prob)
print('accuracy score :')
print(acc * 100)

# build the confusion matrix
conf_matrix = confusion_matrix(y_test, y_prob)
print('Confusion Matrix :')
print(conf_matrix)

auc = roc_auc_score(y_test, y_probs)
mcc = matthews_corrcoef(y_test, y_prob)

print(f"AUC-ROC Score: {auc:.4f}")
print(f"MCC Score:     {mcc:.4f}")

# Classification Report - precision, recall

print("\nFull Report:")
print('Classification Report: ')
print(classification_report(y_test, y_prob))


## a. Problem statement
Develop below Machine Learning binary Classification models for the same dataset.
1. Logistic Regression
2. Decision Tree Classifi er
3. K-Nearest Neighbor Classifi er
4. Naive Bayes Classifier - Gaussian or Multinomial
5. Ensemble Model - Random Forest
6. Ensemble Model - XGBoost

and compute Evaluation metrics in python for all the above models.
1. Accuracy
2. AUC Score
3. Precision
4. Recall
5. F1 Score
6. Matthews Correlation Coeffi cient (MCC Score)


## b. Dataset description
Took Cancer data set for a binary classification is taken from Kaggle site. which has
Feature Size: 30 
Instance Size: 569

## c. Comparison Table with the evaluation metrics calculated for all the 6 models

ML Model Name				|	Accuracy	AUC		Precision	Recall	F1		MCC
----------------------------------------------------------------------------------------------------------
Logistic Regression			|	0.93	0.9782		0.93	0.93		0.93	0.8524
Decision Tree				|	0.97	0.9645		0.97	0.97		0.97	0.9254
kNN							|	0.94	0.974		0.94	0.94		0.94	0.8801
Naive Bayes(Gaussian)		|	0.93	0.9885		0.93	0.93		0.93	0.8501
Random Forest (Ensemble)	|	0.96	0.9953		0.97	0.96		0.96	0.9253
XGBoost (Ensemble)			|	0.96	0.9937		0.96	0.96		0.96	0.9112

## observations on the performance of each model on the chosen dataset

ML Model Name				Observation about model performance
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Logistic Regression:		The high AUC and F1-score suggest this model is effective, particularly if the classes are balanced. These metrics suggest the model is well-suited for applications requiring high reliability in identifying both positive and negative cases.
 
Decision Tree:				The Decision Tree outperforms Logistic Regression in nearly every category, showing a significant lead in Accuracy (97% vs 93%). The Decision Tree has a much higher MCC (0.9254), indicating it is more reliable and robust in its predictions across both classes.

kNN:						The kNN model performs better than Logistic Regression across all metrics (except AUC) but still falls short of the Decision Tree's high marks.

Naive Bayes:				Naive Bayes achieves the highest AUC (0.9885) of all models tested. This indicates it is the most capable of distinguishing between classes across 		various thresholds, even though its overall accuracy is lower than the Decision Tree.

Random Forest (Ensemble):	Naive Bayes achieves the highest AUC (0.9885) of all models tested. This indicates it is the most capable of distinguishing between classes across 		various thresholds, even though its overall accuracy is lower than the Decision Tree.

XGBoost (Ensemble):			XGBoost ties with Random Forest for Accuracy (0.96) and performs exceptionally well across the scoreboard.
# main

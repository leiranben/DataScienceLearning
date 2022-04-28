# Learning how to implement classifier models using the Wisconsin Breast Cancer Dataset
# Dataset is linked here: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic) 


# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


# Loading some data from the Wisconsin Breast Cancer dataset
data = load_breast_cancer(as_frame=True)
df = data.frame
# features = data.feature_names
features = ["worst concave points", "worst radius",  "mean concave points","worst perimeter"]
# Testing some basic statistics
# print(df.describe())

# Trying to predict targets based on the ten parameters given
X_train, X_test, y_train, y_test = train_test_split(data.data[["worst concave points", 
                                                              "worst radius", 
                                                              "mean concave points",
                                                              "worst perimeter"]], data.target, test_size = 0.2, random_state = 42)

# Logistic Regression
# print("Logsitic Regression Model")
reg_log = LogisticRegression()
reg_log.fit(X_train, y_train)
y_pred_log = reg_log.predict(X_test)
# print(metrics.classification_report(y_test, y_pred_log))
# print("roc_auc_score: ", metrics.roc_auc_score(y_test, y_pred_log))
# print("f1 score: ", metrics.f1_score(y_test, y_pred_log))

# Random Forest Classifier Model
# print("Random Forest Classifier Model")
reg_rf = RandomForestClassifier()
reg_rf.fit(X_train, y_train)
y_pred_rf = reg_rf.predict(X_test)
# print(metrics.classification_report(y_test, y_pred_rf))
# print("roc_auc_score: ", metrics.roc_auc_score(y_test, y_pred_rf))
# print("f1 score: ", metrics.f1_score(y_test, y_pred_rf))

# Looking at the importance of different features in the model
feature_df = pd.DataFrame({'Importance':reg_rf.feature_importances_, 'Features': features })
feature_df = feature_df.sort_values(by=["Importance"])
print(feature_df)

# Support Vector Machine Model
# print("SVM Model")
reg_svm = SVC()
reg_svm.fit(X_train, y_train)
y_pred_svm = reg_svm.predict(X_test)
# print(metrics.classification_report(y_test, y_pred_svm))
# print("roc_auc_score: ", metrics.roc_auc_score(y_test, y_pred_svm))
# print("f1 score: ", metrics.f1_score(y_test, y_pred_svm))

# K-Nearest Neighbors Model
# print("K-Nearest Neighbors Model")
reg_knn = KNeighborsClassifier()
reg_knn.fit(X_train, y_train)
y_pred_knn = reg_knn.predict(X_test)
# print(metrics.classification_report(y_test,y_pred_knn))
# print("roc_auc_score: ", metrics.roc_auc_score(y_test, y_pred_svm))
# print("f1 score: ", metrics.f1_score(y_test, y_pred_svm))

# Summarizing the success of various models
summary_df = pd.DataFrame({'Model':["Logistic Regression", "Random Forest", "SVM", "K-Nearest Neighbors"],
                            'ROC':[metrics.roc_auc_score(y_test, y_pred_log),
                                    metrics.roc_auc_score(y_test, y_pred_rf),
                                    metrics.roc_auc_score(y_test, y_pred_svm),
                                    metrics.roc_auc_score(y_test, y_pred_knn)],
                            'F1':[metrics.f1_score(y_test, y_pred_log),
                                    metrics.f1_score(y_test, y_pred_rf),
                                    metrics.f1_score(y_test, y_pred_svm),
                                    metrics.f1_score(y_test, y_pred_knn)]}).sort_values(by=["ROC"])
print(summary_df)
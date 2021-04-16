import pandas as pd
import math
import sklearn
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

#takes predictions and y_test
#returns accuracy of predictions (kNN, SVM, Cross Val, Boostrapping)
def accuracy(predictions, y_test):
    return np.mean(predictions == y_test.to_numpy())

#Converting diagnosis column
#to numerical values
#   1 = M or 0 = B
def reclassify(diagnosis):
    if diagnosis == "M":
        return 1
    else:
        return 0
cancer_df = pd.read_csv("breast_cancer.csv")

cancer_df = cancer_df.drop(columns=["id", "Unnamed: 32"]) #drop id and Unnamed: 32 columns
cancer_df["diagnosis"] = cancer_df["diagnosis"].apply(reclassify) #reclassify B and M in diagnosis column

#print("Shape for cancer_df", cancer_df.shape) #(569, 31)

X = cancer_df.loc[:, cancer_df.columns != 'diagnosis'] #get all columns except for diagnosis
y = cancer_df["diagnosis"] #diagnosisis only y

train_proportion = math.floor(cancer_df.shape[0] * 0.75)

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size = 0.25, random_state = 42)

X_train = X[:train_proportion]
y_train = y[:train_proportion]

X_test = X[train_proportion:]
y_test = y[train_proportion:]

#print(cancer_df.describe()) # to describe all the data
#print(X)
#print(y)

#K-nearest Neighbours

model = KNN(n_neighbors=5)

knn = model.fit(X_train, y_train)

#score = model.score(X_test, y_test)

#print(score) Accuracy of our classification

for i in range(1, 15):
     model = KNN(n_neighbors = i)
     model.fit(X_train, y_train)
     score = knn.score(X_test, y_test)
     y_pred = model.predict(X_test)
     print("Accuracy for k = " + str(i) + ": " , metrics.accuracy_score(y_test, y_pred))

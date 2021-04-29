import pandas as pd
import math
import sklearn
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt


def boostrap(B, data, n_neighbhors):
    accuracies = []

    indices = np.random.randint(0, len(data), (B, len(data)))

    for i in indices:
        cancer_df = data.iloc[i] #entire random sample

        X = cancer_df.loc[:, cancer_df.columns != 'diagnosis']  # get all columns except for diagnosis
        y = cancer_df["diagnosis"]  # diagnosisis only y

        train_proportion = math.floor(cancer_df.shape[0] * 0.75)

        X_train = X[:train_proportion]
        y_train = y[:train_proportion]
        #
        X_test = X[train_proportion:]
        y_test = y[train_proportion:]

        knn = KNN(n_neighbors=n_neighbhors)
        knn.fit(X_train, y_train)

        pred = knn.predict(X_test)
        accu = accuracy(pred, y_test)

        accuracies.append(accu)

    return np.array(accuracies).mean()

#takes predictions and y_test
#returns accuracy of predictions (kNN, SVM, Cross Val, Boostrapping)
def accuracy(predictions, y_test):
    return np.mean(predictions == y_test.to_numpy())

def cross_valKnn(k_folds, data, n_neighbhors):
    accuracies = []
    n, d = data.shape

    for i in range(k_folds):
        X = data.loc[:, data.columns != 'diagnosis']  # get all columns except for diagnosis
        y = data["diagnosis"]  # diagnosisis only y

        T = range(int(math.floor(n * i / k_folds)), int((math.floor(n * (i + 1) / k_folds)) - 1) + 1)

        S = np.arange(n)
        S = np.in1d(S, T)
        S = np.where(~S)[0]  # n array - T

        X_train = X.iloc[S]
        X_test = X.iloc[~S]

        y_train = y.iloc[S]
        y_test = y.iloc[~S]

        knn = KNN(n_neighbors=n_neighbhors)
        knn.fit(X_train, y_train)

        pred = knn.predict(X_test)
        accu = accuracy(pred, y_test)

        accuracies.append(accu)

    return np.array(accuracies).mean()

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

# print("Shape for cancer_df", cancer_df.shape) #(569, 31)

X = cancer_df.loc[:, cancer_df.columns != 'diagnosis'] #get all columns except for diagnosis
y = cancer_df["diagnosis"] #diagnosisis only y

train_proportion = math.floor(cancer_df.shape[0] * 0.75)

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size = 0.25, random_state = 42)

X_train = X[:train_proportion]
y_train = y[:train_proportion]

X_test = X[train_proportion:]
y_test = y[train_proportion:]

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
     print("Accuracy for k = " + str(i) + ": " , accuracy(y_pred, y_test))


def svm(data):
    C_parameter = [1, 5, 10]
    predictions = []
    accurate = [0, 0, 0]
    X_train, X_test, y_train, y_test = train_test_split(data, test_size=0.1, random_state=1)
    for i in range (0, 3):
        model = SVC(C=C_parameter[i])
        model.fit(X_train, y_train)  # hyperplane
        decisions = model.decision_function(X_test)
        predictions[i] = model.predict(X_test)
        accurate[i] = accuracy_score(y_test, predictions)

    most_accurate = accurate.index(max(accurate))
    return predictions[most_accurate]
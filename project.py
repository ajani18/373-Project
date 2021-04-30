import pandas as pd
import math
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.svm import SVC
from sklearn import svm
import matplotlib.pyplot as plt


def boostrap(B, data, n_neighbhor, c_vals, alg):
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

        if alg == "kNN":
            knn = KNN(n_neighbors=n_neighbhor)
            knn.fit(X_train, y_train)

            pred = knn.predict(X_test)
            accu = accuracy(pred, y_test)
            accuracies.append(accu)

        else:
            print("tet")
            model = SVC(C=c_vals)
            model.fit(X_train, y_train)  # hyperplane

            y_pred = model.predict(X_test)
            accu = accuracy(y_pred, y_test)

            accuracies.append(accu)

    return np.array(accuracies).mean()

#takes predictions and y_test
#returns accuracy of predictions (kNN, SVM, Cross Val, Boostrapping)
def accuracy(predictions, y_test):
    return np.mean(predictions == y_test.to_numpy())

# kfolds - (5 or 10) data = data frame
def cross_val(k_folds, data, n_neighbhors,C_vals, alg):
    accuracies = []
    n, d = data.shape

    X = data.loc[:, data.columns != 'diagnosis']  # get all columns except for diagnosis
    y = data["diagnosis"]  # diagnosisis only y

    for i in range(k_folds):
        T = range(int(math.floor(n * i / k_folds)), int((math.floor(n * (i + 1) / k_folds)) - 1) + 1)

        S = np.arange(n)
        S = np.in1d(S, T)

        train_index = np.where(~S)[0]  # n array - T
        test_index = S

        X_train = X.iloc[train_index]
        X_test = X.iloc[test_index]

        y_train = y.iloc[train_index]
        y_test = y.iloc[test_index]

        if alg == "kNN":
            knn = KNN(n_neighbors=n_neighbhors)
            knn.fit(X_train, y_train)

            pred = knn.predict(X_test)
            accu = accuracy(pred, y_test)

            accuracies.append(accu)
        else:
            model = SVC(C= C_vals)
            model.fit(X_train, y_train)  # hyperplane

            y_pred = model.predict(X_test)
            accu = accuracy(y_pred, y_test)

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

X = cancer_df.loc[:, cancer_df.columns != 'diagnosis'] #get all columns except for diagnosis
y = cancer_df["diagnosis"] #diagnosisis only y

train_proportion = math.floor(cancer_df.shape[0] * 0.75)

X_train = X[:train_proportion]
y_train = y[:train_proportion]

X_test = X[train_proportion:]
y_test = y[train_proportion:]

np.random.seed(0)
#K-nearest Neighbours/Cross-Val/Boostrap/Hyperparam

#Hyperparameter w/ Cross-Validation
# k_values = np.arange(1, 15)
# #
# knn_cv = []
# for neigh in k_values:
#     accu = cross_val(10, cancer_df, neigh, 0, "kNN")
#     knn_cv.append(accu)
#
# print("Accruacy from bootstrapping for kNN is", boostrap(30, cancer_df, 14, 0, "kNN"))
# print("The best hyperparameter for SVM by bootstrapping is", max(knn_cv))

# boostrap_knn = []
# for neigh in k_values:
#     accu = boostrap(30, cancer_df, neigh)
#     boostrap_knn.append(accu)

#plot for hyper parameter accuracy
# fig = plt.figure()
# plt.plot(k_values, knn_cv)
# plt.xlabel('k in kNN')
# plt.ylabel('CV-Accuracy')
# fig.suptitle('kNN hyperparameter (k)', fontsize=20)
# plt.show()

#***************SVM/Cross-Val/Boostrap/Hyperparam***************#
# c_values = [1, 5, 10]
# #
# cv_scores = []
# for c in c_values:
#     accu = cross_val(10, cancer_df, 0, c, "SVM")
#     cv_scores.append(accu)
#
# print("Accuracy from bootstrapping for SVM is", boostrap(30, cancer_df, 0, 5, "SVM"))
# print("The best accuracy for SVM by bootstrapping is", max(cv_scores))

# boostrap_score = [] #affirm the accuracy of our predictions we are making
# for c in c_values:
#     accu = boostrap(30, cancer_df, 0, c, "SVM")
#     boostrap_score.append((accu, c))
#

#plot hyperparameter tuning SVM
# fig = plt.figure()
# plt.plot(c_values, cv_scores)
# plt.xlabel('c in SVM')
# plt.ylabel('CV-Accuracy')
# fig.suptitle('SVM hyperparameter (C)', fontsize=20)
# plt.show()

#subset analysis
subset = [100, 200, 300, 400]
acuracy_svm = []
acuracy_kNN = []

# for s in subset:
#     df = cancer_df[:s]
#     train_proportion = math.floor(df.shape[0] * 0.75)
#
#     X_train = X[:train_proportion]
#     y_train = y[:train_proportion]
#
#     X_test = X[train_proportion:]
#     y_test = y[train_proportion:]
#
#     model = SVC(C=5)
#     model.fit(X_train, y_train)  # hyperplane
#
#     y_pred = model.predict(X_test)
#     accu = accuracy(y_pred, y_test)
#
#     acuracy_svm.append(accu)

# fig = plt.figure()
# plt.plot(subset, acuracy_svm)
# plt.xlabel('Subset Size')
# plt.ylabel('Accuracy (%)')
# fig.suptitle('SVM Subset Accuracy', fontsize=20)
# plt.show()

# for s in subset:
#     df = cancer_df[:s]
#     train_proportion = math.floor(df.shape[0] * 0.75)
#
#     X_train = X[:train_proportion]
#     y_train = y[:train_proportion]
#
#     X_test = X[train_proportion:]
#     y_test = y[train_proportion:]
#
#     knn = KNN(n_neighbors=14)
#     knn.fit(X_train, y_train)
#
#     pred = knn.predict(X_test)
#     accu = accuracy(pred, y_test)
#
#     acuracy_kNN.append(accu)

# fig = plt.figure()
# plt.plot(subset, acuracy_kNN)
# plt.xlabel('Subset Size')
# plt.ylabel('Accuracy (%)')
# fig.suptitle('kNN Subset Accuracy', fontsize=20)
# plt.show()
import matplotlib.pyplot as plt
from matplotlib.image import imread
import numpy as np
from numpy import random
from utils import load_data1, split_cross_validation, load_kfold
from sklearn.manifold import TSNE
from mpl_toolkits import mplot3d
import pandas as pd
from sklearn import linear_model
from LogRegression import LogRegression

def plotscatterDataset1(x, y):
    x1 = x[:,0]
    x2 = x[:,1]
    colormap = {0:'red',1:'blue'}
    fig, ax = plt.subplots()
    for i in np.unique(y):
        index = np.where(y == i)
        ax.scatter(x1[index], x2[index], c = colormap[i], label = i, s = 100)
    ax.legend()
    plt.show()

#part a) Code to display scatter plot of dataset_1
x, y = load_data1()
z = np.concatenate((x, y[:,None]),axis=1) 
# print(z[0:5,:])

x_df = pd.DataFrame.from_records(z)
# print(x_df.head())
# print(type(x))
# print(type(y))
plotscatterDataset1(x, y)

#Shuffling
x_df = x_df.sample(frac=1, random_state=1)

#Performing 5-fold cross validation
nfolds = 5
xsplit = split_cross_validation(x_df, nfolds)
df1 = pd.DataFrame(xsplit[0]) 
df2 = pd.DataFrame(xsplit[1]) 
df3 = pd.DataFrame(xsplit[2]) 
df4 = pd.DataFrame(xsplit[3]) 
df5 = pd.DataFrame(xsplit[4]) 
#Adding all the folds to a list
dflist = []
dflist.append(df1)
dflist.append(df2)
dflist.append(df3)
dflist.append(df4)
dflist.append(df5)

#Partb) Using the regression class, find training and validation MSE for each fold
print("Running Logistic Regression model using LogRegression class .. ")
arr = np.arange(0,nfolds)

train_acc1 = []
test_acc1 = []
train_acc2 = []
test_acc2 = []
train_acc3 = []
test_acc3 = []
index=0
for k in range(nfolds):
    x_train, y_train, x_test, y_test = load_kfold(dflist, arr, index)

    index = index+1

    print("Fold ",k+1)
    y_train_np = y_train.to_numpy()
    y_train_np = y_train_np.reshape(y_train.shape[0],1)
    y_test_np = y_test.to_numpy()
    y_test_np = y_test_np.reshape(y_test.shape[0],1)

    print("Running LogRegression .. ")
    model = LogRegression(x_train, y_train_np, x_test, y_test_np, learningrate=0.1,  numiterations=2000, regularization=None, numclasses=2)
    w, b, costlist = model.fit(l2=0.0)
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    train_accuracy = 100 - np.mean(np.abs(y_train_pred - y_train_np)) * 100
    test_accuracy = 100 - np.mean(np.abs(y_test_pred - y_test_np)) * 100
    print("Training Accuracy: {} %".format(train_accuracy))
    print("Testing Accuracy: {} %".format(test_accuracy))
    train_acc1.append(train_accuracy)
    test_acc1.append(test_accuracy)

    print("Running LogRegression with L2 regularization .. ")
    model = LogRegression(x_train, y_train_np, x_test, y_test_np, learningrate=0.1,  numiterations=2000, regularization="l2", numclasses=2)
    #Performing grid search on lambda parameter for finding its optimal value
    optimal_lambda = model.performGridSearch()
    print("After grid search, optimal lambda obtained is ",optimal_lambda)
    w, b, costlist = model.fit(l2=optimal_lambda)
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    train_accuracy = 100 - np.mean(np.abs(y_train_pred - y_train_np)) * 100
    test_accuracy = 100 - np.mean(np.abs(y_test_pred - y_test_np)) * 100
    print("Training Accuracy with regularization: {} %".format(train_accuracy))
    print("Testing Accuracy with regularization: {} %".format(test_accuracy))
    train_acc2.append(train_accuracy)
    test_acc2.append(test_accuracy)

    print("Running sklearn implementation .. ")
    sklearnmodel = linear_model.LogisticRegression(random_state = 42,max_iter= 150)
    train_accuracy = sklearnmodel.fit(x_train, y_train).score(x_train, y_train)
    test_accuracy = sklearnmodel.fit(x_train, y_train).score(x_test, y_test)
    print("Training Accuracy: {} ".format(train_accuracy))
    print("Testing Accuracy: {} ".format(test_accuracy))
    train_acc3.append(train_accuracy)
    test_acc3.append(test_accuracy)
    print("===============================================================================================")
train_acc1 = np.array(train_acc1)
test_acc1 = np.array(test_acc1)
train_acc2 = np.array(train_acc2)
test_acc2 = np.array(test_acc2)
train_acc3 = np.array(train_acc3)
test_acc3 = np.array(test_acc3)
print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$4")
print("Mean Training Accuracy (Logistic Regression) : ",train_acc1.mean())
print("Mean Testing Accuracy (Logistic Regression) : ",test_acc1.mean())

print("Mean Training Accuracy (Logistic Regression with l2 regularization) : ",train_acc2.mean())
print("Mean Testing Accuracy (Logistic Regression with l2 regularization) : ",test_acc2.mean())

print("Mean Training Accuracy (Sklearn Logistic Regression) : ",train_acc3.mean())
print("Mean Testing Accuracy (Sklearn Logistic Regression) : ",test_acc3.mean())
    


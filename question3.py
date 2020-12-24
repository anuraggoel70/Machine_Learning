import matplotlib.pyplot as plt
from matplotlib.image import imread
import numpy as np
from numpy import random
from utils import load_data2, split_cross_validation, load_kfold, classwiseaccuracy
from sklearn.manifold import TSNE
from mpl_toolkits import mplot3d
import pandas as pd
from sklearn import linear_model
from LogRegression import LogRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.metrics import accuracy_score
import joblib

def plotscatterDataset2(x, y):
    x1 = x[:,0]
    x2 = x[:,1]
    colormap = {0:'red',1:'blue',2:'green',3:'yellow'}
    fig, ax = plt.subplots()
    for i in np.unique(y):
        index = np.where(y == i)
        ax.scatter(x1[index], x2[index], c = colormap[i], label = i, s = 100)
    ax.legend()
    plt.show()

#part a) Code to display scatter plot of dataset_2
x, y = load_data2()
z = np.concatenate((x, y),axis=1) 
# print(z[0:5,:])

x_df = pd.DataFrame.from_records(z)
# print(x_df.head())
# print(type(x))
# print(type(y))
plotscatterDataset2(x, y.reshape(y.shape[0],))

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

# Partb) Using the regression class, find training and validation MSE for each fold
print("Running Logistic Regression model using LogRegression class .. ")
arr = np.arange(0,nfolds)

train_acc1 = []
test_acc1 = []
train_acc2 = []
test_acc2 = []
train_acc3 = []
test_acc3 = []
train_acc4 = []
test_acc4 = []
train_acc5 = []
test_acc5 = []
train_acc6 = []
test_acc6 = []
index=0
n_classes = 4
for k in range(nfolds):
    x_train, y_train, x_test, y_test = load_kfold(dflist, arr, index)
    # print(np.unique(y_train, return_counts=True))
    # print(np.unique(y_test, return_counts=True))
    index = index+1

    print("*********************** Fold {} ******************************".format(k+1))
    y_train_np = y_train.to_numpy()
    y_train_np = y_train_np.reshape(y_train.shape[0],1)
    y_test_np = y_test.to_numpy()
    y_test_np = y_test_np.reshape(y_test.shape[0],1)

    print("Running OVO Approach without Regularization .. .. ")

    model = LogRegression(x_train, y_train_np, x_test, y_test_np, learningrate=0.05,  numiterations=2000, regularization=None, numclasses=4)
    w, b = model.multiclassificaton_ovo(l2=0.0)
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    train_accuracy = accuracy_score(y_train_np, y_train_pred)
    test_accuracy = accuracy_score(y_test_np, y_test_pred)
    print("Training Accuracy: {} %".format(train_accuracy))
    print("Testing Accuracy: {} %".format(test_accuracy))
    train_acc1.append(train_accuracy)
    test_acc1.append(test_accuracy)
    print("Classwise accuracy ..")
    for i in range(n_classes):
        print("Class {} accuracy is {} %".format(i,classwiseaccuracy(y_test_pred,y_test_np,i)))

    filename = "ovo_withoutl2_fold" + str(k) + ".sav"
    joblib.dump(model, filename)
 
    loaded_model = joblib.load(filename)
    y_test_pred = loaded_model.predict(x_test)
    print("Loaded model accuracy on test set ", accuracy_score(y_test_np, y_test_pred))

    print("======================================================================================")

    print("Running OVO Approach with Regularization .. .. ")

    model = LogRegression(x_train, y_train_np, x_test, y_test_np, learningrate=0.05,  numiterations=2000, regularization='l2', numclasses=4)
    w, b = model.multiclassificaton_ovo(l2=0.01)
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    train_accuracy = accuracy_score(y_train_np, y_train_pred)
    test_accuracy = accuracy_score(y_test_np, y_test_pred)
    print("Training Accuracy: {} %".format(train_accuracy))
    print("Testing Accuracy: {} %".format(test_accuracy))
    train_acc2.append(train_accuracy)
    test_acc2.append(test_accuracy)
    print("Classwise accuracy ..")
    for i in range(n_classes):
        print("Class {} accuracy is {} %".format(i,classwiseaccuracy(y_test_pred,y_test_np,i)))

    filename = "ovo_withl2_fold" + str(k) + ".sav"
    joblib.dump(model, filename)
 
    loaded_model = joblib.load(filename)
    y_test_pred = loaded_model.predict(x_test)
    print("Loaded model accuracy on test set ", accuracy_score(y_test_np, y_test_pred))

    print("======================================================================================")

    print("Running OVR Approach without Regularization .. .. ")

    model = LogRegression(x_train, y_train_np, x_test, y_test_np, learningrate=0.05,  numiterations=2000, regularization=None, numclasses=4)
    w, b = model.multiclassificaton(l2=0.0)
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    train_accuracy = accuracy_score(y_train_np, y_train_pred)
    test_accuracy = accuracy_score(y_test_np, y_test_pred)
    print("Training Accuracy: {} %".format(train_accuracy))
    print("Testing Accuracy: {} %".format(test_accuracy))
    train_acc3.append(train_accuracy)
    test_acc3.append(test_accuracy)
    print("Classwise accuracy ..")
    for i in range(n_classes):
        print("Class {} accuracy is {} %".format(i,classwiseaccuracy(y_test_pred,y_test_np,i)))

    filename = "ovr_withoutl2_fold" + str(k) + ".sav"
    joblib.dump(model, filename)
 
    loaded_model = joblib.load(filename)
    y_test_pred = loaded_model.predict(x_test)
    print("Loaded model accuracy on test set ", accuracy_score(y_test_np, y_test_pred))

    print("======================================================================================")

    print("Running OVR Approach with Regularization .. .. ")

    model = LogRegression(x_train, y_train_np, x_test, y_test_np, learningrate=0.05,  numiterations=2000, regularization='l2', numclasses=4)
    w, b = model.multiclassificaton(l2=0.01)
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    train_accuracy = 100 - np.mean(np.abs(y_train_pred - y_train_np)) * 100
    test_accuracy = 100 - np.mean(np.abs(y_test_pred - y_test_np)) * 100
    print("Training Accuracy with l2 regularization: {} %".format(train_accuracy))
    print("Testing Accuracy with l2 regularization: {} %".format(test_accuracy))
    train_acc4.append(train_accuracy)
    test_acc4.append(test_accuracy)
    print("Classwise accuracy ..")
    for i in range(n_classes):
        print("Class {} accuracy is {} %".format(i,classwiseaccuracy(y_test_pred,y_test_np,i)))

    filename = "ovr_withl2_fold" + str(k) + ".sav"
    joblib.dump(model, filename)
 
    loaded_model = joblib.load(filename)
    y_test_pred = loaded_model.predict(x_test)
    print("Loaded model accuracy on test set ", accuracy_score(y_test_np, y_test_pred))

    print("Running sklearn implementation of OVO approach.. ")
    sklearnmodel = OneVsOneClassifier(linear_model.LogisticRegression(random_state = 42,max_iter= 150))
    train_accuracy = sklearnmodel.fit(x_train, y_train).score(x_train, y_train)
    test_accuracy = sklearnmodel.fit(x_train, y_train).score(x_test, y_test)
    print("Training Accuracy: {} ".format(train_accuracy))
    print("Testing Accuracy: {} ".format(test_accuracy))
    train_acc5.append(train_accuracy)
    test_acc5.append(test_accuracy)

    print("Running sklearn implementation of OVR approach.. ")
    sklearnmodel = linear_model.LogisticRegression(random_state = 42,max_iter= 150, multi_class='ovr')
    train_accuracy = sklearnmodel.fit(x_train, y_train).score(x_train, y_train)
    test_accuracy = sklearnmodel.fit(x_train, y_train).score(x_test, y_test)
    print("Training Accuracy: {} ".format(train_accuracy))
    print("Testing Accuracy: {} ".format(test_accuracy))
    train_acc6.append(train_accuracy)
    test_acc6.append(test_accuracy)

train_acc1 = np.array(train_acc1)
test_acc1 = np.array(test_acc1)
train_acc2 = np.array(train_acc2)
test_acc2 = np.array(test_acc2)
train_acc3 = np.array(train_acc3)
test_acc3 = np.array(test_acc3)
train_acc4 = np.array(train_acc4)
test_acc4 = np.array(test_acc4)
train_acc5 = np.array(train_acc5)
test_acc5 = np.array(test_acc5)
train_acc6 = np.array(train_acc6)
test_acc6 = np.array(test_acc6)
print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$4")
print("Mean Training Accuracy OVO Approach (Logistic Regression) : ",train_acc1.mean())
print("Mean Testing Accuracy OVO Approach (Logistic Regression) : ",test_acc1.mean())

print("Mean Training Accuracy OVO Approach  (Logistic Regression with l2 regularization) : ",train_acc2.mean())
print("Mean Testing Accuracy OVO Approach (Logistic Regression with l2 regularization) : ",test_acc2.mean())

print("Mean Training Accuracy OVO Approach (Sklearn Logistic Regression) : ",train_acc5.mean())
print("Mean Testing Accuracy OVO Approach (Sklearn Logistic Regression) : ",test_acc5.mean())

print("Mean Training Accuracy OVR Approach (Logistic Regression) : ",train_acc3.mean())
print("Mean Testing Accuracy  OVR Approach (Logistic Regression) : ",test_acc3.mean())

print("Mean Training Accuracy OVR Approach (Logistic Regression with l2 regularization) : ",train_acc4.mean())
print("Mean Testing Accuracy  OVR Approach (Logistic Regression with l2 regularization) : ",test_acc4.mean())

print("Mean Training Accuracy OVR Approach (Sklearn Logistic Regression) : ",train_acc6.mean())
print("Mean Testing Accuracy  OVR Approach (Sklearn Logistic Regression) : ",test_acc6.mean())
    


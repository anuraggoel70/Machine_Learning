from random import randrange
from sklearn.metrics import mean_squared_error
import numpy as np
from scipy.io import loadmat
import pandas as pd
from sklearn.metrics import accuracy_score
import random

def load_data1(data_path='dataset_1.mat'):
    data = loadmat(data_path)
    #print(data.keys())
    x = data['samples']
    y = data['labels']

    y = np.transpose(y)
    y = y.reshape(y.shape[0],)

    # print(x.shape) #5000*2
    # print(y.shape) #5000
    # print(np.unique(y)) #0,1
    return x, y

def load_data2(data_path='dataset_2.mat'):
    data = loadmat(data_path)
    #print(data.keys())
    x = data['samples']
    y = data['labels']

    y = np.transpose(y)
    #y = y.reshape(y.shape[0],)

    # print(x.shape) #10000*2
    # print(y.shape) #10000*1
    # print(np.unique(y)) #0,1,2,3
    # print(np.unique(y, return_counts=True))
    return x, y

def load_dataset1():
    x = pd.read_csv('Regression_data/Dataset.data', delimiter=' ', skiprows=0)
    # print("X shape ",x.shape)  4176*9
    return x

def split_cross_validation(dataset, folds=5):
    dataset_split = list()
    dataset_copy = dataset.values.tolist()
    fold_size = int(len(dataset)/folds)
    count=0
    for i in range(folds):
        fold = list()
        while len(fold)<fold_size:
            fold.append(dataset_copy[count])
            count=count+1
        dataset_split.append(fold)
    return dataset_split

def load_kfold(dflist, arr, index):
    df0 = pd.DataFrame()
    xtest = pd.DataFrame()

    for j in arr:
        if j!=index:
            df0 = pd.concat([df0,dflist[j]],axis=0)
    xtest=pd.concat([xtest,dflist[index]],axis=0) 

    ytrain = df0.iloc[:,2]
    xtrain = df0.iloc[:,0:2]

    ytest = xtest.iloc[:,2]
    xtest = xtest.iloc[:,0:2]

    return xtrain, ytrain, xtest, ytest

def load_kfold_regressiondata(dflist, arr, index):
    df0 = pd.DataFrame()
    xtest = pd.DataFrame()

    for j in arr:
        if j!=index:
            df0 = pd.concat([df0,dflist[j]],axis=0)
    xtest=pd.concat([xtest,dflist[index]],axis=0) 

    ytrain = df0.iloc[:,8]
    xtrain = df0.iloc[:,0:8]

    ytest = xtest.iloc[:,8]
    xtest = xtest.iloc[:,0:8]

    return xtrain, ytrain, xtest, ytest

def mse(ytest, ypred):
    result = np.square(np.subtract(ytest,ypred)).mean()
    return result

def mse_sklearn(ytest, ypred):
    result = mean_squared_error(ytest, ypred)
    return result

def relabelling_data(y, label):
    """
    Relabels data for one-vs-all classifier
    Parameters
        y: m x 1 vector of labels
        label: which label to set to 1 (sets others to 0)
    Returns
        relabeled_data: m x 1 vector of relabeled data
    """
    relabeled_data = np.zeros(len(y))
    for i, val in enumerate(y):
        if val == label:
            relabeled_data[i] = 1
    return relabeled_data.reshape(len(y), 1)

def indices(l, val):
   retval = np.where(l==val)
   return retval

def classwiseaccuracy(y_pred, y_true, class_):
    ytrue = indices(y_true, class_)[0]
    # print(len(ypred))
    # print(len(ytrue))
    count = 0
    for k in range(len(ytrue)):
        if y_pred[ytrue[k]]==class_:
            count = count+1
    return (count/len(ytrue)*100)
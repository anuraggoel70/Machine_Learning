from utils import load_dataset1, split_cross_validation, mse, mse_sklearn, load_kfold_regressiondata
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd
import random
class Regression(object):
    def __init__(self, xtrain, ytrain, xtest, ytest):
        super(Regression, self).__init__()
        self.x = xtrain
        self.y = ytrain
        self.xtest = xtest
        self.ytest = ytest
    def fit(self):
        reg = LinearRegression().fit(self.x, self.y)
        #print(reg.score(self.xtest,self.ytest))
        return reg

    def predictWithNormal(self):
        x_bias = np.ones((len(self.x),1))
        xtrain = np.append(x_bias,self.x,axis=1)

        x_transpose = np.transpose(xtrain)   #calculating transpose
        x_transpose_dot_x = x_transpose.dot(xtrain)  # calculating dot product
        temp_1 = np.linalg.inv(x_transpose_dot_x) #calculating inverse
        temp_2 = x_transpose.dot(self.y)
        theta = temp_1.dot(temp_2) 

        ypred = []
        for i in range(len(self.x)):
            yhat = theta[0]
            for j in range(1,len(theta)):
                yhat = yhat + self.x.iloc[i,j-1]*theta[j]
            ypred.append(yhat)
        ypred = np.array(ypred)
        msecal_train = mse(self.y, ypred)
        msesk_train = mse_sklearn(self.y, ypred)

        ypred = []
        for i in range(len(xtest)):
            yhat = theta[0]
            for j in range(1,len(theta)):
                yhat = yhat + xtest.iloc[i,j-1]*theta[j]
            ypred.append(yhat)
        ypred = np.array(ypred)
        msecal_test = mse(self.ytest, ypred)
        msesk_test = mse(self.ytest, ypred)

        return msecal_train, msesk_train, msecal_test, msesk_test

    def predictimpl(self):
        model = self.fit()
        intercept, params = model.intercept_, model.coef_

        ypred = []
        for i in range(len(self.x)):
            yhat = intercept
            for j in range(len(params)):
                yhat = yhat + self.x.iloc[i,j]*params[j]
            ypred.append(yhat)
        ypred = np.array(ypred)
        msecal_train = mse(self.y, ypred)
        msesk_train = mse_sklearn(self.y, ypred)

        ypred = []
        for i in range(len(self.xtest)):
            yhat = intercept
            for j in range(len(params)):
                yhat = yhat + self.xtest.iloc[i,j]*params[j]
            ypred.append(yhat)
        ypred = np.array(ypred)
        msecal_test = mse(self.ytest, ypred)
        msesk_test = mse(self.ytest, ypred)

        return msecal_train, msesk_train, msecal_test, msesk_test

    def predictWithsklearn(self):
        model = self.fit()

        ypred = model.predict(self.x)
        msecal_train = mse(self.y, ypred)
        msesk_train = mse_sklearn(self.y, ypred)

        ypred = model.predict(self.xtest)
        msecal_test = mse(self.ytest, ypred)
        msesk_test = mse(self.ytest, ypred)

        return msecal_train, msesk_train, msecal_test, msesk_test

#Loading dataset
x = load_dataset1()
#Mapping string valued features to numeric values
x.iloc[:,0] = x.iloc[:,0].map({'F':0, 'I':1, 'M':2})
#Shuffling
x = x.sample(frac=1, random_state=1)
#Performing 5-fold cross validation
nfolds = 5
xsplit = split_cross_validation(x, nfolds)
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
print("Running Regression model using Regression class .. ")
arr = np.arange(0,nfolds)

trainerr = []
trainerr_sklearn = []
testerr = []
testerr_sklearn = []
index=0
for k in range(nfolds):
    xtrain, ytrain, xtest, ytest = load_kfold_regressiondata(dflist, arr, index)

    #Predictions for xtrain and xtest using sklearn  linear regression fit()
    msecal_train, msesk_train, msecal_test, msesk_test = Regression(xtrain, ytrain, xtest, ytest).predictimpl()

    trainerr.append(msecal_train)
    trainerr_sklearn.append(msesk_train)
    testerr.append(msecal_test)
    testerr_sklearn.append(msesk_test)

    print("Fold : ",k+1)
    print("==========================================")
    print("Training error (Calculated) ",msecal_train)
    print("Training error (sklearn) ",msesk_train)
    print("Testing error (Calculated) ",msecal_test)
    print("Testing error (sklearn) ",msesk_test)

    index= index+1

trainerr = np.array(trainerr)
testerr = np.array(testerr)
trainerr_sklearn = np.array(trainerr_sklearn)
testerr_sklearn = np.array(testerr_sklearn)
print("===============================================")
print("Mean Training error (Calculated) : ",trainerr.mean())
print("Mean Testing error (Calculated) : ",testerr.mean())

print("Mean Training error (sklearn) : ",trainerr_sklearn.mean())
print("Mean Testing error (sklearn) : ",testerr_sklearn.mean())



#Partc) Using the Normal Equations Method, find training and validation MSE for each fold
print("Running Regression model using Normal Equation .. ")
arr = np.arange(0,nfolds)

trainerr = []
trainerr_sklearn = []
testerr = []
testerr_sklearn = []
index=0
for k in range(nfolds):
    xtrain, ytrain, xtest, ytest = load_kfold_regressiondata(dflist, arr, index)

    #Predictions for xtrain and xtest using sklearn  linear regression fit()
    msecal_train, msesk_train, msecal_test, msesk_test = Regression(xtrain, ytrain, xtest, ytest).predictWithNormal()

    trainerr.append(msecal_train)
    trainerr_sklearn.append(msesk_train)
    testerr.append(msecal_test)
    testerr_sklearn.append(msesk_test)

    print("Fold : ",k+1)
    print("############################################")
    print("Training error (Calculated) ",msecal_train)
    print("Training error (sklearn) ",msesk_train)
    print("Testing error (Calculated) ",msecal_test)
    print("Testing error (sklearn) ",msesk_test)

    index= index+1

trainerr = np.array(trainerr)
testerr = np.array(testerr)
trainerr_sklearn = np.array(trainerr_sklearn)
testerr_sklearn = np.array(testerr_sklearn)
print("################################################")
print("Mean Training error (Calculated) : ",trainerr.mean())
print("Mean Testing error (Calculated) : ",testerr.mean())

print("Mean Training error (sklearn) : ",trainerr_sklearn.mean())
print("Mean Testing error (sklearn) : ",testerr_sklearn.mean())



#Partd) Using the skleran predictions(), find training and validation MSE for each fold
print("Running Regression model using sklearn predictions .. ")
arr = np.arange(0,nfolds)

trainerr = []
trainerr_sklearn = []
testerr = []
testerr_sklearn = []
index=0
for k in range(nfolds):
    xtrain, ytrain, xtest, ytest = load_kfold_regressiondata(dflist, arr, index)

    #Predictions for xtrain and xtest using sklearn  linear regression fit()
    msecal_train, msesk_train, msecal_test, msesk_test = Regression(xtrain, ytrain, xtest, ytest).predictWithsklearn()

    trainerr.append(msecal_train)
    trainerr_sklearn.append(msesk_train)
    testerr.append(msecal_test)
    testerr_sklearn.append(msesk_test)

    print("Fold : ",k+1)
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    print("Training error (Calculated) ",msecal_train)
    print("Training error (sklearn) ",msesk_train)
    print("Testing error (Calculated) ",msecal_test)
    print("Testing error (sklearn) ",msesk_test)

    index= index+1

trainerr = np.array(trainerr)
testerr = np.array(testerr)
trainerr_sklearn = np.array(trainerr_sklearn)
testerr_sklearn = np.array(testerr_sklearn)
print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$4")
print("Mean Training error (Calculated) : ",trainerr.mean())
print("Mean Testing error (Calculated) : ",testerr.mean())

print("Mean Training error (sklearn) : ",trainerr_sklearn.mean())
print("Mean Testing error (sklearn) : ",testerr_sklearn.mean())

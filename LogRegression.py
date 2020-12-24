import numpy as np
import matplotlib.pyplot as plt
from utils import relabelling_data
class LogRegression(object):
    def __init__(self, xtrain, ytrain, xtest, ytest, learningrate, numiterations, regularization, numclasses):
        super(LogRegression, self).__init__()
        self.xtrain = xtrain
        self.ytrain = ytrain
        self.xtest = xtest
        self.ytest = ytest
        self.learningrate = learningrate
        self.numiterations = numiterations
        self.weights = np.full((xtrain.shape[1],1), 0.01)
        self.bias = 0
        self.numclasses = numclasses
        self.multiclassificatonflag = None

    def sigmoid(self, z):
        y_hat = 1/(1 + np.exp(-z))
        return y_hat

    def performGridSearch(self):
        lamdalist = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]
        optlambda = 1e-1
        bestacc = 0.0
        for value in lamdalist:
            w, b, costlist = self.fit(l2 = value)
            y_test_pred = self.predict(self.xtest)
            test_accuracy = 100 - np.mean(np.abs(y_test_pred - self.ytest)) * 100
            if float(test_accuracy)>float(bestacc):
                bestacc = test_accuracy
                optlambda = value
            # print("Test accuracy {} with lamda {}".format(test_accuracy, value))
        return optlambda

    def multiclassificaton(self, l2):
        if(self.numclasses > 2):
            ytrain_org = self.ytrain
            ytest_org = self.ytest
            weightlist = []
            biaslist = []
            costlist = []
            for i in range(self.numclasses):
                print("Working for Class ", i)
                self.ytrain = relabelling_data(self.ytrain, i)
                self.ytest = relabelling_data(self.ytest, i)
                weight, bias, costl = self.fit(l2=l2)
                weightlist.append(weight)
                biaslist.append(bias)
                costlist.append(costl)
                self.ytrain = ytrain_org
                self.ytest = ytest_org
            self.weights = weightlist
            self.bias = biaslist     
            self.multiclassificatonflag = 'ovr' 
            return self.weights, self.bias

    def multiclassificaton_ovo(self, l2):
        if(self.numclasses > 2):
            xtrain_org = self.xtrain
            xtest_org = self.xtest
            ytrain_org = self.ytrain
            ytest_org = self.ytest
            weightlist = []
            biaslist = []
            costlist = []
            for i in range(self.numclasses):
                for j in range(i+1,self.numclasses):
                    print("***************** i = {} j = {} ****************".format(i,j))
                    #select subset of dataset which contains class labels as i or j
                    train_index_tuple = np.where((ytrain_org==i) | (ytrain_org==j))
                    train_index = train_index_tuple[0]
                    test_index_tuple = np.where((ytest_org==i) | (ytest_org==j))
                    test_index = test_index_tuple[0]
                    xtrain_cur = []
                    ytrain_cur = []
                    for k in train_index:
                        xtrain_cur.append(self.xtrain.iloc[k,:])
                        if self.ytrain[k,:]==i:
                            ytrain_cur.append(0)
                        elif self.ytrain[k,:]==j:
                            ytrain_cur.append(1)
                    xtrain_cur = np.array(xtrain_cur)
                    ytrain_cur = np.array(ytrain_cur)
                    xtest_cur = []
                    ytest_cur = []
                    for k in test_index:
                        xtest_cur.append(self.xtest.iloc[k,:])
                        if self.ytest[k,:]==i:
                            ytest_cur.append(0)
                        elif self.ytest[k,:]==j:
                            ytest_cur.append(1)
                    xtest_cur = np.array(xtest_cur)
                    ytest_cur = np.array(ytest_cur)

                    self.xtrain = xtrain_cur
                    self.xtest = xtest_cur
                    self.ytrain = ytrain_cur.reshape(ytrain_cur.shape[0],1)
                    self.ytest = ytest_cur.reshape(ytest_cur.shape[0],1)

                    # print("Length of xtrain ",len(self.xtrain))
                    # print("Length of xtest ",len(self.xtest))
                    # print("Length of ytrain ",(self.ytrain.shape))
                    # print("Length of ytest ",(self.ytest.shape))

                    self.weights = np.full((self.xtrain.shape[1],1), 0.01)

                    weight, bias, costl = self.fit(l2=l2)
                    weightlist.append(weight)
                    biaslist.append(bias)
                    costlist.append(costl)

                    self.xtrain = xtrain_org
                    self.xtest = xtest_org
                    self.ytrain = ytrain_org
                    self.ytest = ytest_org
            self.weights = weightlist
            self.bias = biaslist 
            self.multiclassificatonflag = 'ovo'     
            return self.weights, self.bias

    def fit(self, l2):
        costlist = []
        losslist_train = []
        acclist_train = []
        losslist_val = []
        acclist_val = []
        iterationlist = []
        w = self.weights
        b = self.bias
        print("lambda value ",l2)
        for i in range(self.numiterations):
            z = np.dot(self.xtrain,w) + b
            yhat = self.sigmoid(z)
            # print("**** iteration {} ******* yhat value {} ".format(i,yhat))
            loss = -((self.ytrain)*(np.log(yhat)))-((1-self.ytrain)*(np.log(1-yhat)))-(l2*np.sum(np.square(w)))
            cost = (np.sum(loss))/self.xtrain.shape[0]
            derivative_weight = np.dot(self.xtrain.T,(yhat-self.ytrain)) - (2*l2*w)
            derivative_weight = derivative_weight/self.xtrain.shape[0]
            derivative_bias = np.sum(yhat-self.ytrain)/self.xtrain.shape[0]
            costlist.append(cost)
            w = w - self.learningrate * derivative_weight
            b = b - self.learningrate * derivative_bias
            if i % 10 == 0:
                losslist_train.append(cost)
                iterationlist.append(i)
                #Calculating validation loss
                z = np.dot(self.xtest,w) + b
                yhat = self.sigmoid(z)
                loss = -((self.ytest)*(np.log(yhat)))-((1-self.ytest)*(np.log(1-yhat)))-(l2*np.sum(np.square(w)))
                cost = (np.sum(loss))/self.xtest.shape[0]
                losslist_val.append(cost)
                #Calculate accuracy on training set
                yhat = self.sigmoid(np.dot(self.xtrain,w)+b)
                ypred = np.zeros((self.xtrain.shape[0],1))
                for i in range(yhat.shape[0]):
                    if yhat[i,0]<= 0.5:
                        ypred[i,0] = 0
                    else:
                        ypred[i,0] = 1
                acc = (100 - np.mean(np.abs(ypred - self.ytrain)) * 100)
                acclist_train.append(acc)
                #Calculate accuracy on validation set
                yhat = self.sigmoid(np.dot(self.xtest,w)+b)
                ypred = np.zeros((self.xtest.shape[0],1))
                for i in range(yhat.shape[0]):
                    if yhat[i,0]<= 0.5:
                        ypred[i,0] = 0
                    else:
                        ypred[i,0] = 1
                acc = (100 - np.mean(np.abs(ypred - self.ytest)) * 100)
                acclist_val.append(acc)
        plt.plot(iterationlist, losslist_train, 'g', label='Training loss')
        plt.plot(iterationlist, losslist_val, 'b', label='Validation loss')
        plt.title('Training and Validation loss')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.legend()
        # plt.show()

        plt.plot(iterationlist, acclist_train, 'g', label='Training accuracy')
        plt.plot(iterationlist, acclist_val, 'b', label='Validation accuracy')
        plt.title('Training and Validation accuracy')
        plt.xlabel('Iterations')
        plt.ylabel('Accuracy')
        plt.legend()
        # plt.show()

        self.weights = w
        self.bias = b
        return self.weights, self.bias, costlist

    def predict(self, xtest):
        if self.multiclassificatonflag is None:
            yhat = self.sigmoid(np.dot(xtest,self.weights)+self.bias)
            ypred = np.zeros((xtest.shape[0],1))
            # if z is bigger than 0.5, our prediction is sign one (y_head=1),
            # if z is smaller than 0.5, our prediction is sign zero (y_head=0),
            for i in range(yhat.shape[0]):
                if yhat[i,0]<= 0.5:
                    ypred[i,0] = 0
                else:
                    ypred[i,0] = 1
        elif self.multiclassificatonflag=='ovr':
            wlist = self.weights
            blist = self.bias
            predictions = np.empty((xtest.shape[0], 0))
            # Gather predictions for each classifier
            for i in range(len(wlist)):
                z = np.dot(xtest, wlist[i]) + blist[i]
                predictions = np.append(predictions, self.sigmoid(z), axis=1)
            ypred = np.argmax(predictions, axis=1).reshape(xtest.shape[0], 1)
        elif self.multiclassificatonflag=='ovo':
            wlist = self.weights
            blist = self.bias
            predictions = []
            xtest = xtest.to_numpy()
            # Gather predictions for each classifier
            
            for k in xtest:
                wlist_index=0
                count = np.zeros((self.numclasses,))
                for i in range(self.numclasses):
                    for j in range(i+1, self.numclasses):
                        p = wlist[wlist_index]
                        q = blist[wlist_index]
                        z = np.dot(k,p) + q 
                        if z>=0:
                            count[j]=count[j]+1
                        else:
                            count[i]=count[i]+1
                        wlist_index = wlist_index+1
                final_prediction = np.argmax(count)        
                predictions.append(final_prediction)
            ypred = np.array(predictions)
        return ypred

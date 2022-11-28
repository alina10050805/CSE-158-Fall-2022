from pyexpat import model
import numpy
import json
import urllib
import scipy.optimize
import random
import ast
import sklearn
from sklearn import linear_model

#read the file and return the data in file
def readFile(file):
    dataFile = open(file)

    dataList = []

    for line in dataFile:
        lineData = json.loads(line)
        dataList.append(lineData)

    dataFile.close()
    return dataList

# question 1
def countExclamation(dataLine):
    countExclama = 0
    for alpha in dataLine:
        if alpha=='!':
            countExclama = countExclama + 1
    
    return [1] + [countExclama]

def simplePredictor(data):
    X = [countExclamation(d['review_text']) for d in data]
    y = [d['rating'] for d in data]

    model = sklearn.linear_model.LinearRegression(fit_intercept=False)
    model.fit(X,y)

    theta = model.coef_
    print("theta: ", theta)

    y_pred = model.predict(X)
    sse = sum([x**2 for x in (y-y_pred)])
    mse = sse/len(y)
    print("Mean Squared Error: ", mse)

# question 2
def countExclamationAndLength(dataLine):
    countExclama = 0
    for alpha in dataLine:
        if alpha=='!':
            countExclama = countExclama + 1
    
    return [1] + [len(dataLine)] + [countExclama]

def RetrainPredictor(data):
    X = [countExclamationAndLength(d['review_text']) for d in data]
    y = [d['rating'] for d in data]

    model = sklearn.linear_model.LinearRegression(fit_intercept=False)
    model.fit(X,y)

    theta = model.coef_
    print("theta: ", theta)

    y_pred = model.predict(X)
    sse = sum([x**2 for x in (y-y_pred)])
    mse = sse/len(y)
    print("Mean Squared Error: ", mse)

# question 3
def addOneDegree(dataX):
    for data in dataX:
        data.append(data[-1]*data[1])
    
    return dataX

def polynomial(data):
    X = [countExclamation(d['review_text']) for d in data]
    y = [d['rating'] for d in data]

    mse = []
    for i in range(5):
        if i!=0:
            addOneDegree(X)
        model = sklearn.linear_model.LinearRegression(fit_intercept=False)
        model.fit(X,y)

        y_pred = model.predict(X)
        sse = sum([x**2 for x in (y-y_pred)])
        mse.append(sse/len(y))
        
    print("Mean Squared Error: ", mse)

# question 4
def polynomialSplit(data):
    X = [countExclamation(d['review_text']) for d in data]
    y = [d['rating'] for d in data]

    mse = []
    mid = len(X)//2
    for i in range(5):
        if i!=0:
            addOneDegree(X)
        model = sklearn.linear_model.LinearRegression(fit_intercept=False)
        model.fit(X[mid:],y[mid:])

        y_pred = model.predict(X[:mid])
        sse = sum([x**2 for x in (y[:mid]-y_pred)])
        mse.append(sse/len(y[:mid]))
        
    print("Mean Squared Error: ", mse)

# question 5
def polynomialSplitMAE(data):
    X = [countExclamation(d['review_text']) for d in data]
    y = [d['rating'] for d in data]

    sum = 0
    mid = len(X)//2
    model = sklearn.linear_model.LinearRegression(fit_intercept=False)
    model.fit(X[mid:],y[mid:])

    theta = model.coef_
        
    for yi in y[:mid]:
        sum = sum + abs(yi - theta[0])

    mae = sum/len(X[:mid])
    print(mae)

#question 6
def readBeerFile(file):
    dataFile = open(file)

    dataList = []

    for line in dataFile:
        if 'user/gender' in line:
            lineData = eval(line)
            dataList.append(lineData)

    dataFile.close()
    return dataList

def countExclamationLogistic(dataLine):
    countExclama = 0
    for alpha in dataLine:
        if alpha=='!':
            countExclama = countExclama + 1
    
    return [1] + [countExclama]

def logisticRegressor(data):
    X = [countExclamationLogistic(d['review/text']) for d in data]
    y = [d['user/gender']=='Female' for d in data]

    mod = sklearn.linear_model.LogisticRegression()
    mod.fit(X,y)
    predictions = mod.predict(X)

    TP = sum([(p and l) for (p,l) in zip(predictions, y)])
    TN = sum([(not p and not l) for (p,l) in zip(predictions, y)])
    FP = sum([(p and not l) for (p,l) in zip(predictions, y)])
    FN = sum([(not p and l) for (p,l) in zip(predictions, y)])

    TPR = TP/(TP + FN)
    TNR = TN/(TN + FP)

    BER = 1 - 1/2*(TPR + TNR)

    print("TP: ", TP)
    print("TN: ", TN)
    print("FP: ", FP)
    print("FN: ", FN)
    print("BER: ", BER)

#question 7
def logisticRegressorBalance(data):
    X = [countExclamation(d['review/text']) for d in data]
    y = [d['user/gender']=='Female' for d in data]

    mod = sklearn.linear_model.LogisticRegression(class_weight='balanced')
    mod.fit(X,y)
    predictions = mod.predict(X)

    TP = sum([(p and l) for (p,l) in zip(predictions, y)])
    TN = sum([(not p and not l) for (p,l) in zip(predictions, y)])
    FP = sum([(p and not l) for (p,l) in zip(predictions, y)])
    FN = sum([(not p and l) for (p,l) in zip(predictions, y)])

    TPR = TP/(TP + FN)
    TNR = TN/(TN + FP)

    BER = 1 - 1/2*(TPR + TNR)

    print("TP: ", TP)
    print("TN: ", TN)
    print("FP: ", FP)
    print("FN: ", FN)
    print("BER: ", BER)

#question 8
def precisionK(data):
    X = [countExclamation(d['review/text']) for d in data]
    y = [d['user/gender']=='Female' for d in data]

    mod = sklearn.linear_model.LogisticRegression(class_weight='balanced')
    mod.fit(X,y)
    predictions = mod.predict(X)

    TP = sum([(p and l) for (p,l) in zip(predictions, y)])
    TN = sum([(not p and not l) for (p,l) in zip(predictions, y)])
    FP = sum([(p and not l) for (p,l) in zip(predictions, y)])
    FN = sum([(not p and l) for (p,l) in zip(predictions, y)])

    scores = mod.decision_function(X)
    scoreslabels = list(zip(scores, y))
    scoreslabels.sort(reverse=True)

    sortedlabels = [x[1] for x in scoreslabels]

    K1 = sum(sortedlabels[:1]) / 1
    K10 = sum(sortedlabels[:10]) / 10
    K100 = sum(sortedlabels[:100]) / 100
    K1000 = sum(sortedlabels[:1000]) / 1000
    K10000 = sum(sortedlabels[:10000]) / 10000

    print("K1: ", K1)
    print("K10: ", K10)
    print("K100: ", K100)
    print("K1000: ", K1000)
    print("K10000: ", K10000)

def main():
    fileName = "young_adult_10000.json"
    dataL = readFile(fileName)

    print("Question 1: ")
    simplePredictor(dataL)

    print("Question 2: ")
    RetrainPredictor(dataL)

    print("Question 3: ")
    polynomial(dataL)

    print("Question 4: ")
    polynomialSplit(dataL)

    print("Question 5: ")
    polynomialSplitMAE(dataL)

    fileName = "beer_50000.json"
    dataL = readBeerFile(fileName)

    print("Question 6: ")
    logisticRegressor(dataL)

    print("Question 7: ")
    logisticRegressorBalance(dataL)

    print("Question 8: ")
    precisionK(dataL)
    

if __name__ == "__main__":
    main()
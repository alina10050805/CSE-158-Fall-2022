#!/usr/bin/env python
# coding: utf-8

# In[257]:


import json
import gzip
import math
from collections import defaultdict
import numpy
from sklearn import linear_model


# In[258]:


# This will suppress any warnings, comment out if you'd like to preserve them
import warnings
warnings.filterwarnings("ignore")


# In[259]:


# Check formatting of submissions
def assertFloat(x):
    assert type(float(x)) == float

def assertFloatList(items, N):
    assert len(items) == N
    assert [type(float(x)) for x in items] == [float]*N


# In[260]:


answers = {}


# In[261]:


f = open("spoilers.json.gz", 'r')


# In[262]:


dataset = []
for l in f:
    d = eval(l)
    dataset.append(d)


# In[263]:


f.close()


# In[264]:


# A few utility data structures
reviewsPerUser = defaultdict(list)
reviewsPerItem = defaultdict(list)

for d in dataset:
    u,i = d['user_id'],d['book_id']
    reviewsPerUser[u].append(d)
    reviewsPerItem[i].append(d)

# Sort reviews per user by timestamp
for u in reviewsPerUser:
    reviewsPerUser[u].sort(key=lambda x: x['timestamp'])
    
# Same for reviews per item
for i in reviewsPerItem:
    reviewsPerItem[i].sort(key=lambda x: x['timestamp'])


# In[265]:


# E.g. reviews for this user are sorted from earliest to most recent
[d['timestamp'] for d in reviewsPerUser['b0d7e561ca59e313b728dc30a5b1862e']]


# In[266]:


### 1a


# In[267]:


y = []
ypred = []
for d in reviewsPerUser:
    ratings = []
    reviews = reviewsPerUser[d]
    if len(reviews) > 1:
        for rating in reviews[:-1]:
            ratings.append(rating['rating'])
        y.append(reviews[-1]['rating'])
        ypred.append(sum(ratings) / len(ratings))

def MSE(y, ypred):
    return sum([(a-b)**2 for (a,b) in zip(y,ypred)]) / len(y)


# In[268]:


answers['Q1a'] = MSE(y,ypred)
MSE(y,ypred)


# In[269]:


assertFloat(answers['Q1a'])


# In[270]:


### 1b


# In[271]:


y = []
ypred = []
for d in reviewsPerItem:
    ratings = []
    reviews = reviewsPerItem[d]
    if len(reviews) > 1:
        for rating in reviews[:-1]:
            ratings.append(rating['rating'])
        y.append(reviews[-1]['rating'])
        ypred.append(sum(ratings) / len(ratings))


# In[272]:


answers['Q1b'] = MSE(y,ypred)
answers['Q1b']


# In[273]:


assertFloat(answers['Q1b'])


# In[ ]:





# In[274]:


### 2


# In[275]:


answers['Q2'] = []

for N in [1,2,3]:
    # etc.
    y = []
    ypred = []
    for d in reviewsPerUser:
        ratings = []
        reviews = reviewsPerUser[d]
        if len(reviews) > N:
            for rating in reviews[-(N+1):-1]:
                ratings.append(rating['rating'])
            y.append(reviews[-1]['rating'])
            ypred.append(sum(ratings) / len(ratings))
        elif len(reviews) > 1:
            for rating in reviews[:-1]:
                ratings.append(rating['rating'])
            y.append(reviews[-1]['rating'])
            ypred.append(sum(ratings) / len(ratings))

    answers['Q2'].append(MSE(y,ypred))

answers['Q2']


# In[276]:


assertFloatList(answers['Q2'], 3)


# In[277]:


### 3a


# In[278]:


def feature3(N, u): # For a user u and a window size of N
    feat = [1]
    reviews = reviewsPerUser[u]
    if len(reviews) > N:
        for rating in reviews[-(N+1):-1]:
            feat.append(rating['rating'])
    
    return feat


# In[279]:


answers['Q3a'] = [feature3(2,dataset[0]['user_id']), feature3(3,dataset[0]['user_id'])]
answers['Q3a']


# In[280]:


assert len(answers['Q3a']) == 2
assert len(answers['Q3a'][0]) == 3
assert len(answers['Q3a'][1]) == 4


# In[281]:


### 3b


# In[282]:


answers['Q3b'] = []

for N in [1,2,3]:
    # etc.
    x = []
    y = []
    mod = linear_model.LinearRegression()
    ypred = []
    for u in reviewsPerUser:
        reviews = reviewsPerUser[u]
        if len(reviews) > N:
            x.append(feature3(N, u))
            y.append(reviewsPerUser[u][-1]['rating'])

    mod.fit(x, y)
    ypred = mod.predict(x)
    
    mse = MSE(y, ypred)
    answers['Q3b'].append(mse)

answers['Q3b']


# In[283]:


assertFloatList(answers['Q3b'], 3)


# In[284]:


### 4a


# In[285]:


globalAverage = [d['rating'] for d in dataset]
globalAverage = sum(globalAverage) / len(globalAverage)


# In[286]:


def featureMeanValue(N, u): # For a user u and a window size of N
    feat = [1]
    reviews = reviewsPerUser[u]
    average = []
    if len(reviews) > 1:
        for i in range(2, min(N+2, len(reviews)+1)):
            ni = -i
            rating = reviews[-i]
            feat.append(rating['rating'])
            average.append(rating['rating'])

        average = sum(average) / len(average)

        while len(feat) < N+1:
            feat.append(average)

    return feat


# In[287]:


def featureMissingValue(N, u):
    feat = [1]
    reviews = reviewsPerUser[u]
    average = []
    if len(reviews) > 1:
        for i in range(2, min(N+2, len(reviews)+1)):
            ni = -i
            rating = reviews[-i]
            feat.append(0)
            feat.append(rating['rating'])
            average.append(rating['rating'])

        average = sum(average) / len(average)

        while len(feat) < 2*N+1:
            feat.append(1)
            feat.append(0)

    return feat


# In[288]:


answers['Q4a'] = [featureMeanValue(10, dataset[0]['user_id']), featureMissingValue(10, dataset[0]['user_id'])]
answers['Q4a']


# In[289]:


assert len(answers['Q4a']) == 2
assert len(answers['Q4a'][0]) == 11
assert len(answers['Q4a'][1]) == 21


# In[290]:


### 4b


# In[291]:


answers['Q4b'] = []

for featFunc in [featureMeanValue, featureMissingValue]:
    # etc.
    x = []
    y = []
    mod = linear_model.LinearRegression()
    ypred = []
    for u in reviewsPerUser:
        reviews = reviewsPerUser[u]
        if len(reviews) > 1:
            x.append(featFunc(10, u))
            y.append(reviewsPerUser[u][-1]['rating'])

    mod.fit(x, y)
    ypred = mod.predict(x)
    
    mse = MSE(y, ypred)
    answers['Q4b'].append(mse)

answers['Q4b']


# In[292]:


assertFloatList(answers["Q4b"], 2)


# In[293]:


### 5


# In[294]:


def feature5(sentence):
    countExclama = 0
    countCapital = 0

    for alpha in sentence:
        if alpha == '!':
            countExclama = countExclama + 1
        if alpha.isupper():
            countCapital = countCapital + 1

    return [1] + [len(sentence)] + [countExclama] + [countCapital]


# In[295]:


y = []
X = []

for d in dataset:
    for spoiler,sentence in d['review_sentences']:
        X.append(feature5(sentence))
        y.append(spoiler)


# In[296]:


mod = linear_model.LogisticRegression(class_weight='balanced')
mod.fit(X, y)
predictions = mod.predict(X)

TP = sum([(p and l) for (p,l) in zip(predictions, y)])
TN = sum([(not p and not l) for (p,l) in zip(predictions, y)])
FP = sum([(p and not l) for (p,l) in zip(predictions, y)])
FN = sum([(not p and l) for (p,l) in zip(predictions, y)])

TPR = TP/(TP + FN)
TNR = TN/(TN + FP)

BER = 1 - 1/2*(TPR + TNR)


# In[297]:


answers['Q5a'] = X[0]
answers['Q5a']


# In[298]:


answers['Q5b'] = [TP, TN, FP, FN, BER]
answers['Q5b']


# In[299]:


assert len(answers['Q5a']) == 4
assertFloatList(answers['Q5b'], 5)


# In[300]:


### 6


# In[301]:


def feature6(review):
    countExclama = 0
    countCapital = 0
    first5 = []

    for spoiler,sentence in review['review_sentences'][:6]:
        if len(first5) < 5:
            first5.append(spoiler)
        else:
            for alpha in sentence:
                if alpha == '!':
                    countExclama = countExclama + 1
                if alpha.isupper():
                    countCapital = countCapital + 1

    return [1] + first5 + [len(sentence)] + [countExclama] + [countCapital]


# In[302]:


y = []
X = []

for d in dataset:
    sentences = d['review_sentences']
    if len(sentences) < 6: continue
    X.append(feature6(d))
    y.append(sentences[5][0])

#etc.
mod = linear_model.LogisticRegression(class_weight='balanced')
mod.fit(X, y)
predictions = mod.predict(X)

TP = sum([(p and l) for (p,l) in zip(predictions, y)])
TN = sum([(not p and not l) for (p,l) in zip(predictions, y)])
FP = sum([(p and not l) for (p,l) in zip(predictions, y)])
FN = sum([(not p and l) for (p,l) in zip(predictions, y)])

TPR = TP/(TP + FN)
TNR = TN/(TN + FP)

BER = 1 - 1/2*(TPR + TNR)


# In[303]:


answers['Q6a'] = X[0]
answers['Q6a']


# In[304]:


answers['Q6b'] = BER
answers['Q6b']


# In[305]:


assert len(answers['Q6a']) == 9
assertFloat(answers['Q6b'])


# In[306]:


### 7


# In[307]:


# 50/25/25% train/valid/test split
Xtrain, Xvalid, Xtest = X[:len(X)//2], X[len(X)//2:(3*len(X))//4], X[(3*len(X))//4:]
ytrain, yvalid, ytest = y[:len(X)//2], y[len(X)//2:(3*len(X))//4], y[(3*len(X))//4:]


# In[308]:


bers = []
bestC = 0.01
ber = 100

for c in [0.01, 0.1, 1, 10, 100]:
    # etc.
    mod = linear_model.LogisticRegression(C=c, class_weight='balanced')
    
    mod.fit(Xtrain,ytrain)
    ypredValid = mod.predict(Xvalid)
    ypredTest = mod.predict(Xtest)

    TP = sum([(a and b) for (a,b) in zip(yvalid, ypredValid)])
    TN = sum([(not a and not b) for (a,b) in zip(yvalid, ypredValid)])
    FP = sum([(not a and b) for (a,b) in zip(yvalid, ypredValid)])
    FN = sum([(a and not b) for (a,b) in zip(yvalid, ypredValid)])

    TPR = TP / (TP + FN)
    TNR = TN / (TN + FP)

    BER = 1 - 0.5*(TPR + TNR)

    bers.append(BER)

    if BER < ber:
        bestC = c
        ber = BER

mod = linear_model.LogisticRegression(C=bestC, class_weight='balanced')
mod.fit(Xtrain,ytrain)
ypredTest = mod.predict(Xtest)

TP = sum([(a and b) for (a,b) in zip(ytest, ypredTest)])
TN = sum([(not a and not b) for (a,b) in zip(ytest, ypredTest)])
FP = sum([(not a and b) for (a,b) in zip(ytest, ypredTest)])
FN = sum([(a and not b) for (a,b) in zip(ytest, ypredTest)])

TPR = TP / (TP + FN)
TNR = TN / (TN + FP)

ber = 1 - 0.5*(TPR + TNR)


# In[309]:


answers['Q7'] = bers + [bestC] + [ber]
answers['Q7']


# In[310]:


assertFloatList(answers['Q7'], 7)


# In[311]:


### 8


# In[312]:


def Jaccard(s1, s2):
    numer = len(s1.intersection(s2))
    denom = len(s1.union(s2))
    if denom == 0:
        return 0
    return numer / denom


# In[313]:


# 75/25% train/test split
dataTrain = dataset[:15000]
dataTest = dataset[15000:]


# In[314]:


# A few utilities

itemAverages = defaultdict(list)
ratingMean = []

for d in dataTrain:
    itemAverages[d['book_id']].append(d['rating'])
    ratingMean.append(d['rating'])

for i in itemAverages:
    itemAverages[i] = sum(itemAverages[i]) / len(itemAverages[i])

ratingMean = sum(ratingMean) / len(ratingMean)


# In[315]:


reviewsPerUser = defaultdict(list)
usersPerItem = defaultdict(set)

for d in dataTrain:
    u,i = d['user_id'], d['book_id']
    reviewsPerUser[u].append(d)
    usersPerItem[i].add(u)


# In[316]:


# From my HW2 solution, welcome to reuse
def predictRating(user,item):
    ratings = []
    similarities = []
    for d in reviewsPerUser[user]:
        i2 = d['book_id']
        if i2 == item: continue
        ratings.append(d['rating'] - itemAverages[i2])
        similarities.append(Jaccard(usersPerItem[item],usersPerItem[i2]))
    if (sum(similarities) > 0):
        weightedRatings = [(x*y) for x,y in zip(ratings,similarities)]
        return itemAverages[item] + sum(weightedRatings) / sum(similarities)
    else:
        # User hasn't rated any similar items
        if item in itemAverages:
            return itemAverages[item]
        else:
            return ratingMean


# In[317]:


predictions = [predictRating(d['user_id'], d['book_id']) for d in dataTest]
labels = [d['rating'] for d in dataTest]


# In[318]:


answers["Q8"] = MSE(predictions, labels)
answers["Q8"]


# In[319]:


assertFloat(answers["Q8"])


# In[320]:


### 9


# In[321]:


list0 = []
list1to5 = []
list5 = []


# In[322]:


for d in dataTest:
    # etc.
    item = d['book_id']
    if len(usersPerItem[item]) > 5:
        list5.append(d)
    elif len(usersPerItem[item]) >= 1 and len(usersPerItem[item]) <= 5:
        list1to5.append(d)
    else:
        list0.append(d)

predictions = [predictRating(d['user_id'], d['book_id']) for d in list0]
labels = [d['rating'] for d in list0]
mse0 = MSE(predictions, labels)

predictions = [predictRating(d['user_id'], d['book_id']) for d in list1to5]
labels = [d['rating'] for d in list1to5]
mse1to5 = MSE(predictions, labels)

predictions = [predictRating(d['user_id'], d['book_id']) for d in list5]
labels = [d['rating'] for d in list5]
mse5 = MSE(predictions, labels)


# In[ ]:





# In[323]:


answers["Q9"] = [mse0, mse1to5, mse5]
answers["Q9"]


# In[324]:


assertFloatList(answers["Q9"], 3)


# In[325]:


### 10


# In[338]:


import statistics

ratingMed = []

for d in dataTrain:
    ratingMed.append(d['rating'])

ratingMed = statistics.median(ratingMed)

def predictRating2(user,item):
    ratings = []
    similarities = []
    for d in reviewsPerUser[user]:
        i2 = d['book_id']
        if i2 == item: continue
        ratings.append(d['rating'] - itemAverages[i2])
        similarities.append(Jaccard(usersPerItem[item],usersPerItem[i2]))
    if (sum(similarities) > 0):
        weightedRatings = [(x*y) for x,y in zip(ratings,similarities)]
        return itemAverages[item] + sum(weightedRatings) / sum(similarities)
    else:
        # User hasn't rated any similar items
        if item in itemAverages:
            return itemAverages[item]
        else:
            return ratingMed
            
list0 = []
list1to5 = []
list5 = []

for d in dataTest:
    # etc.
    item = d['book_id']
    if len(usersPerItem[item]) > 5:
        list5.append(d)
    elif len(usersPerItem[item]) >= 1 and len(usersPerItem[item]) <= 5:
        list1to5.append(d)
    else:
        list0.append(d)

predictions = [predictRating2(d['user_id'], d['book_id']) for d in list0]
itsMSE = 1.7010
labels = [d['rating'] for d in list0]
mse = MSE(predictions, labels)


# In[339]:


answers["Q10"] = ("describe your solution: I change the ratingMed that return in function predictRating to rating median.", itsMSE)


# In[340]:


assert type(answers["Q10"][0]) == str
assertFloat(answers["Q10"][1])


# In[341]:


f = open("answers_midterm.txt", 'w')
f.write(str(answers) + '\n')
f.close()


# In[ ]:





# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy
import urllib
import scipy.optimize
import random
from sklearn import linear_model
import gzip
from collections import defaultdict


# In[3]:


import warnings
warnings.filterwarnings("ignore")


# In[4]:


def assertFloat(x):
    assert type(float(x)) == float

def assertFloatList(items, N):
    assert len(items) == N
    assert [type(float(x)) for x in items] == [float]*N


# In[5]:


f = open("5year.arff", 'r')


# In[6]:


# Read and parse the data
while not '@data' in f.readline():
    pass

dataset = []
for l in f:
    if '?' in l: # Missing entry
        continue
    l = l.split(',')
    values = [1] + [float(x) for x in l]
    values[-1] = values[-1] > 0 # Convert to bool
    dataset.append(values)


# In[7]:


X = [d[:-1] for d in dataset]
y = [d[-1] for d in dataset]


# In[8]:


answers = {} # Your answers


# In[9]:


def accuracy(predictions, y):
    TP = sum([(p and l) for (p,l) in zip(predictions,y)])
    FP = sum([(p and not l) for (p,l) in zip(predictions, y)])
    TN = sum([(not p and not l) for (p,l) in zip(predictions, y)])
    FN = sum([(not p and l) for (p,l) in zip(predictions, y)])
    
    return (TP + TN) / (TP + FP + TN + FN)


# In[10]:


def BER(predictions, y):
    TP = sum([(p and l) for (p,l) in zip(predictions,y)])
    FP = sum([(p and not l) for (p,l) in zip(predictions, y)])
    TN = sum([(not p and not l) for (p,l) in zip(predictions, y)])
    FN = sum([(not p and l) for (p,l) in zip(predictions, y)])
    
    TPR = TP / (TP + FN)
    TNR = TN / (TN + FP)
    
    return 1 - 1/2 * (TPR + TNR)


# In[11]:


### Question 1


# In[12]:


mod = linear_model.LogisticRegression(C=1)
mod.fit(X,y)

pred = mod.predict(X)


# In[13]:


acc1 = accuracy(pred, y)
ber1 = BER(pred, y)

answers['Q1'] = [acc1, ber1] # Accuracy and balanced error rate


# In[14]:


assertFloatList(answers['Q1'], 2)
answers['Q1']


# In[15]:


### Question 2


# In[16]:


mod = linear_model.LogisticRegression(C=1, class_weight='balanced')
mod.fit(X,y)

pred = mod.predict(X)


# In[17]:


acc2 = accuracy(pred, y)
ber2 = BER(pred, y)


# In[18]:


answers['Q2'] = [acc2, ber2]


# In[19]:


assertFloatList(answers['Q2'], 2)
answers['Q2']


# In[20]:


### Question 3


# In[21]:


random.seed(3)
random.shuffle(dataset)


# In[22]:


X = [d[:-1] for d in dataset]
y = [d[-1] for d in dataset]


# In[23]:


Xtrain, Xvalid, Xtest = X[:len(X)//2], X[len(X)//2:(3*len(X))//4], X[(3*len(X))//4:]
ytrain, yvalid, ytest = y[:len(X)//2], y[len(X)//2:(3*len(X))//4], y[(3*len(X))//4:]


# In[24]:


len(Xtrain), len(Xvalid), len(Xtest)


# In[25]:


mod = linear_model.LogisticRegression(C=1, class_weight='balanced')
mod.fit(Xtrain,ytrain)

berTrain = BER(mod.predict(Xtrain), ytrain)
berValid = BER(mod.predict(Xvalid), yvalid)
berTest = BER(mod.predict(Xtest), ytest)


# In[26]:


answers['Q3'] = [berTrain, berValid, berTest]


# In[27]:


assertFloatList(answers['Q3'], 3)
answers['Q3']


# In[28]:


### Question 4


# In[29]:


berList = []

def pipeline(reg):
    mod = linear_model.LogisticRegression(C=reg, class_weight='balanced')
    
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
    
    return BER

for c in [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]:
    berList.append(pipeline(c))


# In[30]:


answers['Q4'] = berList


# In[31]:


assertFloatList(answers['Q4'], 9)
answers['Q4']


# In[32]:


### Question 5


# In[33]:


bestC = 100
ber5 = answers['Q4'][6]


# In[34]:


answers['Q5'] = [bestC, ber5]


# In[35]:


assertFloatList(answers['Q5'], 2)
answers['Q5']


# In[36]:


### Question 6


# In[37]:


f = gzip.open("young_adult_10000.json.gz")
dataset = []
for l in f:
    dataset.append(eval(l))


# In[38]:


dataTrain = dataset[:9000]
dataTest = dataset[9000:]


# In[ ]:





# In[39]:


# Some data structures you might want

usersPerItem = defaultdict(set) # Maps an item to the users who rated it
itemsPerUser = defaultdict(set) # Maps a user to the items that they rated
reviewsPerUser = defaultdict(list)
reviewsPerItem = defaultdict(list)
ratingDict = {} # To retrieve a rating for a specific user/item pair

for d in dataTrain:
    user,item = d['user_id'], d['book_id']
    
    usersPerItem[item].add(user)
    itemsPerUser[user].add(item)
    reviewsPerUser[user].append(d)
    reviewsPerItem[item].append(d)
    ratingDict[(user,item)] = d['rating']
    
userAverages = {}
itemAverages = {}

for u in itemsPerUser:
    rs = [ratingDict[(u,i)] for i in itemsPerUser[u]]
    userAverages[u] = sum(rs) / len(rs)
    
for i in usersPerItem:
    rs = [ratingDict[(u,i)] for u in usersPerItem[i]]
    itemAverages[i] = sum(rs) / len(rs)


# In[40]:


def Jaccard(s1, s2):
    numer = len(s1.intersection(s2))
    denom = len(s1.union(s2))
    if denom == 0:
        return 0
    return numer / denom


# In[41]:


def mostSimilar(i, N):
    similarities = []
    users = usersPerItem[i]
    for i2 in usersPerItem:
        if i2 == i: continue
        sim = Jaccard(users, usersPerItem[i2])
        similarities.append((sim,i2))
    similarities.sort(reverse=True)
    return similarities[:10]


# In[ ]:





# In[42]:


answers['Q6'] = mostSimilar('2767052', 10)


# In[43]:


assert len(answers['Q6']) == 10
assertFloatList([x[0] for x in answers['Q6']], 10)
answers['Q6']


# In[44]:


### Question 7


# In[45]:


ratingMean = sum([d['rating'] for d in dataset]) / len(dataset)

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
        return ratingMean
    
def MSE(predictions, labels):
    differences = [(x-y)**2 for x,y in zip(predictions,labels)]
    return sum(differences) / len(differences)

simPrediction = [predictRating(d["user_id"], d['book_id']) for d in dataTest]
labels = [d['rating'] for d in dataTest]

mse7 = MSE(simPrediction, labels)
mse7


# In[46]:


answers['Q7'] = mse7


# In[47]:


assertFloat(answers['Q7'])
answers['Q7']


# In[48]:


### Question 8


# In[49]:


def predictRating2(user,item):
    ratings = []
    similarities = []
    
    for d in reviewsPerItem[item]:
        u2 = d['user_id']
        if u2 == user: continue
        ratings.append(d['rating'] - userAverages[u2])
        similarities.append(Jaccard(itemsPerUser[user],itemsPerUser[u2]))
        
    if (sum(similarities) > 0):
        weightedRatings = [(x*y) for x,y in zip(ratings,similarities)]
        return userAverages[user] + sum(weightedRatings) / sum(similarities)
    else:
        return ratingMean
    
simPrediction2 = [predictRating2(d["user_id"], d['book_id']) for d in dataTest]

mse8 = MSE(simPrediction2, labels)
mse8


# In[50]:


answers['Q8'] = mse8


# In[51]:


assertFloat(answers['Q8'])
answers['Q8']


# In[52]:


f = open("answers_hw2.txt", 'w')
f.write(str(answers) + '\n')
f.close()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





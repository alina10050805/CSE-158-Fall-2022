#!/usr/bin/env python
# coding: utf-8

# In[360]:


import gzip
from collections import defaultdict
import math
import scipy.optimize
from sklearn import svm
import numpy
import string
import random
import string
from sklearn import linear_model
from nltk.stem.porter import *
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords


# ## Read prediction(both classes)

# In[361]:


def readCSV(path):
    f = gzip.open(path, 'rt')
    f.readline()
    for l in f:
        u,b,r = l.strip().split(',')
        r = int(r)
        yield u,b,r


# In[362]:


allRatings = []
for l in readCSV("train_Interactions.csv.gz"):
    allRatings.append(l)

len(allRatings)


# In[363]:


ratingsTrain = allRatings[:190000]
ratingsValid = allRatings[190000:]
ratingsPerUser = defaultdict(list)
ratingsPerItem = defaultdict(list)
for u,b,r in ratingsTrain:
    ratingsPerUser[u].append((b,r))
    ratingsPerItem[b].append((u,r))


# In[364]:


# Copied from baseline code
bookCount = defaultdict(int)
totalRead = 0

for user,book,_ in readCSV("train_Interactions.csv.gz"):
    bookCount[book] += 1
    totalRead += 1

mostPopular = [(bookCount[x], x) for x in bookCount]
mostPopular.sort()
mostPopular.reverse()

totalRead


# In[365]:


valid = [[d[0], d[1], 1] for d in ratingsValid]

books = set()
booksPerUser = defaultdict(set)
usersPerBook = defaultdict(set)
readSet = set()
for u,b,r in readCSV("train_Interactions.csv.gz"):
    usersPerBook[b].add(u)
    booksPerUser[u].add(b)
    books.add(b)
    readSet.add((u, b))

notRead = set()
for d in ratingsValid:
    b = random.choice(list(books))
    while (d[0], b) in readSet or (d[0], b) in notRead:
        b = random.choice(list(books))
    notRead.add((d[0], b))
    # noReadBooks = books - booksPerUser[d[0]]
    # randomBook = random.choice(list(noReadBooks))
    # valid.append([d[0], randomBook, 0])

for u,b in notRead:
    valid.append([u, b, 0])


# In[366]:


acc2 = 0
thresholdPopular = 0
for thre in range(1, 101):
    return1 = set()
    count = 0
    for ic, i in mostPopular:
        count += ic
        return1.add(i)
        if count > totalRead * thre * 0.01: break

    prediction = 0
    for d in valid:
        if d[1] in return1:
            prediction += (d[2]==1)
        else:
            prediction += (d[2]==0)
    
    accThre = prediction / len(valid)
 
    if accThre > acc2:
        acc2 = accThre
        thresholdPopular = thre * 0.01

print(acc2)
print(thresholdPopular)


# In[367]:


return1 = set()
count = 0
for ic, i in mostPopular:
    count += ic
    return1.add(i)
    if count > totalRead * thresholdPopular: break


# In[368]:


def Jaccard(s1, s2):
    numer = len(s1.intersection(s2))
    denom = len(s1.union(s2))
    if denom > 0:
        return numer / denom
    return 0

def mostSimilar(u, b):
    maxSim = 0
    books = booksPerUser[u]
    for book in books:
        if book == b: continue
        sim = Jaccard(usersPerBook[b], usersPerBook[book])
        if sim > maxSim:
            maxSim = sim
            
    return maxSim


# In[369]:


acc = 0
thresholdJaccard = 0
for thre in range(1, 11):
    prediction = 0
    for d in valid:
        maxSimilar = mostSimilar(d[0], d[1])
        
        # if len(similarList)==0: continue
        
        if maxSimilar >= thre * 0.1 or d[1] in return1:
            prediction += (d[2]==1)
        else:
            prediction += (d[2]==0)
            
    accThre = prediction / len(valid)
 
    if accThre > acc:
        acc = accThre
        thresholdJaccard = thre * 0.1

print(acc)
print(thresholdJaccard)


# In[370]:


# write the predictions data in the file
predictions = open("predictions_Read.csv", 'w')
for l in open("pairs_Read.csv"):
    # write the label to the first line of file
    if l.startswith("userID"):
        predictions.write(l)
        continue
    u,b = l.strip().split(',')
    
    maxSimilar = mostSimilar(u, b)

    # if len(similarList)==0:
    #     predictions.write(u + "," + b + "," + "0\n")
    #     continue

    if b in return1 or maxSimilar >= thresholdJaccard:
        predictions.write(u + "," + b + "," + "1\n")
    else:
        predictions.write(u + "," + b + "," + "0\n")

predictions.close()


# ## Category prediction (CSE158 only)

# In[3]:


def readGz(path):
    for l in gzip.open(path, 'rt'):
        yield eval(l)


# In[4]:


data = []

for d in readGz("train_Category.json.gz"):
    data.append(d)


# In[5]:


wordCount = defaultdict(int)
punctuation = set(string.punctuation)
stemmer = PorterStemmer()

stop = set(stopwords.words('english'))
for d in data:
  r = ''.join([c for c in d['review_text'].lower() if not c in punctuation])
  for w in r.split():
    w = stemmer.stem(w)
    
    if w in stop: continue
    wordCount[w] += 1

counts = [(wordCount[w], w) for w in wordCount]
counts.sort()
counts.reverse()
len(wordCount)


# In[6]:


words = [x[1] for x in counts[:40000]]
wordId = dict(zip(words, range(len(words))))
wordSet = set(words)

def feature(datum):
    feat = [0]*len(words)
    r = ''.join([c for c in datum['review_text'].lower() if not c in punctuation])
    
    for w in r.split():
        w = stemmer.stem(w)
        if w in words:
            feat[wordId[w]] += 1
    feat.append(1) #offset
    return feat

X = [feature(d) for d in data]
y = [d['genreID'] for d in data]


# In[7]:


X_train = X
y_train = y
# X_train = X[:9*len(X)//10]
# y_train = y[:9*len(y)//10]
# X_valid = X[9*len(X)//10:]
# y_valid = y[9*len(y)//10:]


# In[8]:


model = linear_model.LogisticRegression(C=0.1)
model.fit(X_train, y_train)
# predictions = model.predict(X_valid)
# correct = predictions == y_valid
# acc8 = sum(correct) / len(correct)
# acc8


# In[9]:


# Run on test set
dataTest = []
for d in readGz("test_Category.json.gz"):
    dataTest.append(d)

Xdata = [feature(d) for d in dataTest]
predData = model.predict(Xdata)
predData


# In[10]:


predictions = open("predictions_Category.csv", 'w')
pos = 0

index = 0
for l in open("pairs_Category.csv"):
    if l.startswith("userID"):
        predictions.write(l)
        continue
    u,b = l.strip().split(',')
    
    predictions.write(u + ',' + b + ',' + str(predData[index]) + '\n')
    index += 1

predictions.close()


# In[ ]:





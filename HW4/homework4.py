#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gzip
import math
import numpy
import random
import sklearn
import string
from collections import defaultdict
from gensim.models import Word2Vec
from nltk.stem.porter import *
from sklearn import linear_model
from sklearn.manifold import TSNE
import dateutil


# In[2]:


import warnings
warnings.filterwarnings("ignore")


# In[3]:


def assertFloat(x):
    assert type(float(x)) == float

def assertFloatList(items, N):
    assert len(items) == N
    assert [type(float(x)) for x in items] == [float]*N


# In[4]:


dataset = []

f = gzip.open("young_adult_20000.json.gz")
for l in f:
    d = eval(l)
    dataset.append(d)
    if len(dataset) >= 20000:
        break
        
f.close()


# In[5]:


answers = {}


# In[6]:


dataset[0]


# In[7]:


### Question 1


# In[8]:


def mostCommonUnigrams():
    wordCount = defaultdict(int)
    punctuation = set(string.punctuation)
    for d in dataset:
        r = ''.join([c for c in d['review_text'].lower() if not c in punctuation])
        ws = r.split()
        for w in ws:
            wordCount[w] += 1

    counts = [(wordCount[w], w) for w in wordCount]
    counts.sort()
    counts.reverse()

    words = [x[1] for x in counts[:1000]]

    wordId = dict(zip(words, range(len(words))))
    wordSet = set(words)

    def feature(datum):
        feat = [0]*len(words)
        r = ''.join([c for c in datum['review_text'].lower() if not c in punctuation])
        ws = r.split()
        for w in ws:
            if w in words:
                feat[wordId[w]] += 1
        feat.append(1) #offset
        return feat

    X = [feature(d) for d in dataset]
    y = [d['rating'] for d in dataset]

    Xtrain = X[:10000]
    Xtext = X[10000:]
    ytrain = y[:10000]
    ytest = y[10000:]

    clf = linear_model.Ridge(1.0, fit_intercept=False) # MSE + 1.0 l2
    clf.fit(Xtrain, ytrain)
    theta = clf.coef_
    predictions = clf.predict(Xtext)

    wordSort = list(zip(theta[:-1], words))
    wordSort.sort()

    mse = sum((ytest - predictions)**2)/len(ytest)

    return mse, wordSort


# In[9]:


def mostCommonBigrams():
    wordCount = defaultdict(int)
    punctuation = set(string.punctuation)
    for d in dataset:
        r = ''.join([c for c in d['review_text'].lower() if not c in punctuation])
        ws = r.split()
        ws2 = [' '.join(x) for x in list(zip(ws[:-1],ws[1:]))]
        for w in ws2:
            wordCount[w] += 1

    counts = [(wordCount[w], w) for w in wordCount]
    counts.sort()
    counts.reverse()

    words = [x[1] for x in counts[:1000]]

    wordId = dict(zip(words, range(len(words))))
    wordSet = set(words)

    def feature(datum):
        feat = [0]*len(words)
        r = ''.join([c for c in datum['review_text'].lower() if not c in punctuation])
        ws = r.split()
        ws2 = [' '.join(x) for x in list(zip(ws[:-1],ws[1:]))]
        for w in ws2:
            if w in words:
                feat[wordId[w]] += 1
        feat.append(1) #offset
        return feat

    X = [feature(d) for d in dataset]
    y = [d['rating'] for d in dataset]

    Xtrain = X[:10000]
    Xtext = X[10000:]
    ytrain = y[:10000]
    ytest = y[10000:]

    clf = linear_model.Ridge(1.0, fit_intercept=False) # MSE + 1.0 l2
    clf.fit(Xtrain, ytrain)
    theta = clf.coef_
    predictions = clf.predict(Xtext)

    wordSort = list(zip(theta[:-1], words))
    wordSort.sort()

    mse = sum((ytest - predictions)**2)/len(ytest)

    return mse, wordSort


# In[10]:


def mostCommonBoth():
    wordCount = defaultdict(int)
    punctuation = set(string.punctuation)
    for d in dataset:
        r = ''.join([c for c in d['review_text'].lower() if not c in punctuation])
        ws = r.split()
        ws2 = [' '.join(x) for x in list(zip(ws[:-1],ws[1:]))]
        for w in ws + ws2:
            wordCount[w] += 1

    counts = [(wordCount[w], w) for w in wordCount]
    counts.sort()
    counts.reverse()

    words = [x[1] for x in counts[:1000]]

    wordId = dict(zip(words, range(len(words))))
    wordSet = set(words)

    def feature(datum):
        feat = [0]*len(words)
        r = ''.join([c for c in datum['review_text'].lower() if not c in punctuation])
        ws = r.split()
        ws2 = [' '.join(x) for x in list(zip(ws[:-1],ws[1:]))]
        for w in ws + ws2:
            if w in words:
                feat[wordId[w]] += 1
        feat.append(1) #offset
        return feat

    X = [feature(d) for d in dataset]
    y = [d['rating'] for d in dataset]

    Xtrain = X[:10000]
    Xtext = X[10000:]
    ytrain = y[:10000]
    ytest = y[10000:]

    clf = linear_model.Ridge(1.0, fit_intercept=False) # MSE + 1.0 l2
    clf.fit(Xtrain, ytrain)
    theta = clf.coef_
    predictions = clf.predict(Xtext)

    wordSort = list(zip(theta[:-1], words))
    wordSort.sort()

    mse = sum((ytest - predictions)**2)/len(ytest)

    return mse, wordSort


# In[11]:


for q,wList in ('Q1a', mostCommonUnigrams), ('Q1b', mostCommonBigrams), ('Q1c', mostCommonBoth):
    mse, wordSort = wList()

    answers[q] = [float(mse), [x[1] for x in wordSort[:5]], [x[1] for x in wordSort[-5:]]]

print(answers['Q1a'])
print(answers['Q1b'])
print(answers['Q1c'])


# In[12]:


for q in 'Q1a', 'Q1b', 'Q1c':
    assert len(answers[q]) == 3
    assertFloat(answers[q][0])
    assert [type(x) for x in answers[q][1]] == [str]*5
    assert [type(x) for x in answers[q][2]] == [str]*5


# In[13]:


### Question 2


# In[14]:


wordCount = defaultdict(int)
punctuation = set(string.punctuation)
for d in dataset:
    r = d['review_text']
    ws = r.split()
    for w in ws:
        wordCount[w] += 1

counts = [(wordCount[w], w) for w in wordCount]
counts.sort()
counts.reverse()

words = [x[1] for x in counts[:1000]]

df = defaultdict(int)
for d in dataset:
    r = d['review_text']
    for w in set(r.split()):
        df[w] += 1

rev = dataset[9]

tf = defaultdict(int)
r = rev['review_text']
for w in r.split():
    # Note = rather than +=, different versions of tf could be used instead
    tf[w] = 1
    
tfidf = dict(zip(words,[tf[w] * math.log2(len(dataset) / df[w]) for w in words]))
tfidfQuery = [tf[w] * math.log2(len(dataset) / df[w]) for w in words]

def Cosine(x1,x2):
    numer = 0
    norm1 = 0
    norm2 = 0
    for a1,a2 in zip(x1,x2):
        numer += a1*a2
        norm1 += a1**2
        norm2 += a2**2
    if norm1*norm2:
        return numer / math.sqrt(norm1*norm2)
    return 0

similarities = []
for rev2 in dataset:
    tf = defaultdict(int)
    r = rev2['review_text']
    for w in r.split():
        # Note = rather than +=
        tf[w] = 1
    tfidf2 = [tf[w] * math.log2(len(dataset) / df[w]) for w in words]
    similarities.append((Cosine(tfidfQuery, tfidf2), rev2['review_text']))

similarities.sort(reverse=True)
sim, review = similarities[0]


# In[15]:


answers['Q2'] = [sim, review]
answers['Q2']


# In[16]:


assert len(answers['Q2']) == 2
assertFloat(answers['Q2'][0])
assert type(answers['Q2'][1]) == str


# In[17]:


### Question 3


# In[18]:


reviewsPerUser = defaultdict(list)


# In[19]:


for d in dataset:
    reviewsPerUser[d['user_id']].append((dateutil.parser.parse(d['date_added']), d['book_id']))


# In[20]:


reviewLists = []
for u in reviewsPerUser:
    rl = list(reviewsPerUser[u])
    rl.sort()
    reviewLists.append([x[1] for x in rl])

model10 = Word2Vec(reviewLists,
                 min_count=1, # Words/items with fewer instances are discarded
                 vector_size=10, # Model dimensionality
                 window=3, # Window size
                 sg=1) # Skip-gram model


# In[21]:


reviewLists[0][0]


# In[22]:


similarities = model10.wv.similar_by_word(reviewLists[0][0])[:5]
similarities


# In[23]:


answers['Q3'] = similarities # probably want model10.wv.similar_by_word(...)[:5]
answers['Q3']


# In[24]:


assert len(answers['Q3']) == 5
assert [type(x[0]) for x in answers['Q3']] == [str]*5
assertFloatList([x[1] for x in answers['Q3']], 5)


# In[25]:


### Question 4


# In[26]:


ratingMean = sum([d['rating'] for d in dataset]) / len(dataset)

bookAverages = defaultdict(list)
reviewsPerUser = defaultdict(list)
    
for d in dataset:
    b = d['book_id']
    u = d['user_id']
    bookAverages[b].append(d['rating'])
    reviewsPerUser[u].append(d)
    
for b in bookAverages:
    bookAverages[b] = sum(bookAverages[b]) / len(bookAverages[b])

def predictRating(user,item):
    ratings = []
    similarities = []
    if not str(item) in model10.wv:
        return ratingMean
    for d in reviewsPerUser[user]:
        i2 = d['book_id']
        if i2 == item: continue
        ratings.append(d['rating'] - bookAverages[i2])
        if str(i2) in model10.wv:
            similarities.append(model10.wv.distance(str(item), str(i2)))
    if (sum(similarities) > 0):
        weightedRatings = [(x*y) for x,y in zip(ratings,similarities)]
        return bookAverages[item] + sum(weightedRatings) / sum(similarities)
    else:
        return ratingMean


# In[27]:


def MSE(predictions, labels):
    differences = [(x-y)**2 for x,y in zip(predictions,labels)]
    return sum(differences) / len(differences)


# In[28]:


predictions = [predictRating(d['user_id'],d['book_id']) for d in dataset]
labels = [d['rating'] for d in dataset]


# In[29]:


mse4 = MSE(predictions, labels)


# In[30]:


answers['Q4'] = mse4
answers['Q4']


# In[31]:


assertFloat(answers['Q4'])


# In[32]:


### Q5


# In[86]:


reviewsPerItem = defaultdict(list)
for d in dataset:
    reviewsPerItem[d['book_id']].append((dateutil.parser.parse(d['date_added']), d['user_id']))


# In[87]:


reviewLists = []
for i in reviewsPerItem:
    rl = list(reviewsPerItem[i])
    rl.sort()
    reviewLists.append([x[1] for x in rl])


# In[88]:


model = Word2Vec(reviewLists,
                 min_count=1, # Words/items with fewer instances are discarded
                 vector_size=10, # Model dimensionality
                 window=3, # Window size
                 sg=1) # Skip-gram model


# In[89]:


reviewLists[0]


# In[90]:


model.wv.similar_by_word(reviewLists[0][0])


# In[91]:


ratingMean = sum([d['rating'] for d in dataset]) / len(dataset)

userAverages = defaultdict(list)
reviewsPerItem = defaultdict(list)
    
for d in dataset:
    b = d['book_id']
    u = d['user_id']
    userAverages[u].append(d['rating'])
    reviewsPerItem[b].append(d)
    
for u in userAverages:
    userAverages[u] = sum(userAverages[u]) / len(userAverages[u])

def predictRating(user,item):
    ratings = []
    similarities = []
    if not str(user) in model.wv:
        return ratingMean
    for d in reviewsPerItem[item]:
        u2 = d['user_id']
        if u2 == user: continue
        ratings.append(d['rating'] - userAverages[u2])
        if str(u2) in model.wv:
            similarities.append(model.wv.distance(str(user), str(u2)))
    if (sum(similarities) > 0):
        weightedRatings = [(x*y) for x,y in zip(ratings,similarities)]
        return userAverages[user] + sum(weightedRatings) / sum(similarities)
    else:
        return ratingMean


# In[92]:


def MSE(predictions, labels):
    differences = [(x-y)**2 for x,y in zip(predictions,labels)]
    return sum(differences) / len(differences)


# In[93]:


predictions = [predictRating(d['user_id'],d['book_id']) for d in dataset]
labels = [d['rating'] for d in dataset]


# In[94]:


mse5 = MSE(predictions, labels)
mse5


# ****

# In[61]:


reviewsPerUser = defaultdict(list)


# In[62]:


for d in dataset:
    reviewsPerUser[d['user_id']].append((dateutil.parser.parse(d['date_added']), d['book_id']))


# In[63]:


reviewLists = []
for u in reviewsPerUser:
    rl = list(reviewsPerUser[u])
    rl.sort()
    reviewLists.append([x[1] for x in rl])

model10 = Word2Vec(reviewLists,
                 min_count=1, # Words/items with fewer instances are discarded
                 vector_size=10, # Model dimensionality
                 window=3, # Window size
                 sg=5) # Skip-gram model


# In[64]:


reviewLists[0]


# In[65]:


similarities = model10.wv.similar_by_word(reviewLists[0][0])[:5]
similarities


# In[66]:


ratingMean = sum([d['rating'] for d in dataset]) / len(dataset)

bookAverages = defaultdict(list)
reviewsPerUser = defaultdict(list)
    
for d in dataset:
    b = d['book_id']
    u = d['user_id']
    bookAverages[b].append(d['rating'])
    reviewsPerUser[u].append(d)
    
for b in bookAverages:
    bookAverages[b] = sum(bookAverages[b]) / len(bookAverages[b])

def predictRating(user,item):
    ratings = []
    similarities = []
    if not str(item) in model10.wv:
        return ratingMean
    for d in reviewsPerUser[user]:
        i2 = d['book_id']
        if i2 == item: continue
        ratings.append(d['rating'] - bookAverages[i2])
        if str(i2) in model10.wv:
            similarities.append(model10.wv.distance(str(item), str(i2)))
    if (sum(similarities) > 0):
        weightedRatings = [(x*y) for x,y in zip(ratings,similarities)]
        return bookAverages[item] + sum(weightedRatings) / sum(similarities)
    else:
        return ratingMean


# In[67]:


def MSE(predictions, labels):
    differences = [(x-y)**2 for x,y in zip(predictions,labels)]
    return sum(differences) / len(differences)


# In[68]:


predictions = [predictRating(d['user_id'],d['book_id']) for d in dataset]
labels = [d['rating'] for d in dataset]


# In[69]:


mse5 = MSE(predictions, labels)
mse5


# In[70]:


answers['Q5'] = ["I change the parameters sg(Skip-gram) of model from 1 to 5. The skip-gram model builds a model by predicting surrounding words given the current word.",
                 mse5]


# In[72]:


assert len(answers['Q5']) == 2
assert type(answers['Q5'][0]) == str
assertFloat(answers['Q5'][1])


# In[73]:


f = open("answers_hw4.txt", 'w')
f.write(str(answers) + '\n')
f.close()


# In[ ]:





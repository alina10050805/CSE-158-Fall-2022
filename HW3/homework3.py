#!/usr/bin/env python
# coding: utf-8

# In[34]:


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


# In[35]:


import warnings
warnings.filterwarnings("ignore")


# In[36]:


def assertFloat(x):
    assert type(float(x)) == float

def assertFloatList(items, N):
    assert len(items) == N
    assert [type(float(x)) for x in items] == [float]*N


# In[37]:


def readGz(path):
    for l in gzip.open(path, 'rt'):
        yield eval(l)


# In[38]:


def readCSV(path):
    f = gzip.open(path, 'rt')
    f.readline()
    for l in f:
        u,b,r = l.strip().split(',')
        r = int(r)
        yield u,b,r


# In[39]:


answers = {}


# In[40]:


# Some data structures that will be useful


# In[41]:


allRatings = []
for l in readCSV("train_Interactions.csv.gz"):
    allRatings.append(l)


# In[42]:


len(allRatings)


# In[43]:


ratingsTrain = allRatings[:190000]
ratingsValid = allRatings[190000:]
ratingsPerUser = defaultdict(list)
ratingsPerItem = defaultdict(list)
for u,b,r in ratingsTrain:
    ratingsPerUser[u].append((b,r))
    ratingsPerItem[b].append((u,r))


# In[ ]:


##################################################
# Rating prediction (CSE258 only)                #
##################################################


# In[ ]:





# In[ ]:


### Question 9


# In[ ]:





# In[ ]:


answers['Q9'] = validMSE


# In[ ]:


assertFloat(answers['Q9'])


# In[ ]:


### Question 10


# In[ ]:


answers['Q10'] = [maxUser, minUser, maxBeta, minBeta]


# In[ ]:


assert [type(x) for x in answers['Q10']] == [str, str, float, float]


# In[ ]:


### Question 11


# In[ ]:





# In[ ]:


answers['Q11'] = (lamb, validMSE)


# In[ ]:


assertFloat(answers['Q11'][0])
assertFloat(answers['Q11'][1])


# In[ ]:


predictions = open("predictions_Rating.csv", 'w')
for l in open("pairs_Rating.csv"):
    if l.startswith("userID"): # header
        predictions.write(l)
        continue
    u,b = l.strip().split(',') # Read the user and item from the "pairs" file and write out your prediction
    # (etc.)
    
predictions.close()


# In[44]:


##################################################
# Read prediction                                #
##################################################


# In[45]:


# Copied from baseline code
bookCount = defaultdict(int)
totalRead = 0

for user,book,_ in readCSV("train_Interactions.csv.gz"):
    bookCount[book] += 1
    totalRead += 1

mostPopular = [(bookCount[x], x) for x in bookCount]
mostPopular.sort()
mostPopular.reverse()

return1 = set()
count = 0
for ic, i in mostPopular:
    count += ic
    return1.add(i)
    if count > totalRead/2: break


# 

# In[46]:


print(allRatings[0])


# In[47]:


### Question 1


# In[48]:


xValid = [[d[0], d[1]] for d in ratingsValid]
yValid = [d[2] for d in ratingsValid]

prediction = []
for d in xValid:
    if d[1] in return1:
        prediction.append(1)
    else:
        prediction.append(0)

predValid = numpy.array(prediction) == numpy.array(yValid)
acc1 = sum(predValid) / len(predValid)


# In[49]:


answers['Q1'] = acc1
answers['Q1']


# In[ ]:


assertFloat(answers['Q1'])


# In[ ]:


### Question 2


# In[ ]:





# In[ ]:


answers['Q2'] = [threshold, acc2]


# In[ ]:


assertFloat(answers['Q2'][0])
assertFloat(answers['Q2'][1])


# In[ ]:


### Question 3/4


# In[ ]:





# In[ ]:


answers['Q3'] = acc3
answers['Q4'] = acc4


# In[ ]:


assertFloat(answers['Q3'])
assertFloat(answers['Q4'])


# In[ ]:


predictions = open("predictions_Read.csv", 'w')
for l in open("pairs_Read.csv"):
    if l.startswith("userID"):
        predictions.write(l)
        continue
    u,b = l.strip().split(',')
    # (etc.)

predictions.close()


# In[ ]:


answers['Q5'] = "I confirm that I have uploaded an assignment submission to gradescope"


# In[ ]:


assert type(answers['Q5']) == str


# In[ ]:


##################################################
# Category prediction (CSE158 only)              #
##################################################


# In[ ]:


### Question 6


# In[ ]:


data = []

for d in readGz("train_Category.json.gz"):
    data.append(d)


# In[ ]:


data[0]


# In[ ]:





# In[ ]:


answers['Q6'] = counts[:10]


# In[ ]:


assert [type(x[0]) for x in answers['Q6']] == [int]*10
assert [type(x[1]) for x in answers['Q6']] == [str]*10


# In[ ]:


### Question 7


# In[ ]:





# In[ ]:


Xtrain = X[:9*len(X)//10]
ytrain = y[:9*len(y)//10]
Xvalid = X[9*len(X)//10:]
yvalid = y[9*len(y)//10:]


# In[ ]:





# In[ ]:


answers['Q7'] = acc7


# In[ ]:


assertFloat(answers['Q7'])


# In[ ]:


### Question 8


# In[ ]:





# In[ ]:


answers['Q8'] = acc8


# In[ ]:


assertFloat(answers['Q8'])


# In[ ]:


# Run on test set


# In[ ]:


predictions = open("predictions_Category.csv", 'w')
pos = 0

for l in open("pairs_Category.csv"):
    if l.startswith("userID"):
        predictions.write(l)
        continue
    u,b = l.strip().split(',')
    # (etc.)

predictions.close()


# In[33]:


f = open("answers_hw3.txt", 'w')
f.write(str(answers) + '\n')
f.close()


# In[ ]:





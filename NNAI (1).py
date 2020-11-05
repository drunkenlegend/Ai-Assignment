#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import pandas as pd
from keras import Sequential
from keras.layers import Dense


# In[9]:


data = pd.read_excel('surprise_and_others.xlsx')


# In[10]:


data.drop(['frame', ' confidence'], axis = 1, inplace = True)
data


# In[11]:


data=data.sample(frac=1,random_state=1)
data


# In[12]:


def preproc(data):
    X = data.iloc[:,:-1].to_numpy()
    Y = data.iloc[:,-1].to_numpy()
    #X_norm = (X - np.mean(X, axis = 0))/np.std(X, axis = 0)
    
    return X, Y


# In[13]:


X_train, Y_train = preproc(data)


# In[14]:


X_train.shape


# Adding layers

# In[15]:


model = Sequential()
#First Hidden Layer
model.add(Dense(100, activation='relu', input_dim=8))
#Second  Hidden Layer
model.add(Dense(92, activation='relu'))
#Third  Hidden Layer
model.add(Dense(61, activation='relu'))
#Output Layer
model.add(Dense(1, activation='sigmoid'))


# Fit, evaluation and prediction
# 

# In[16]:


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=100, batch_size=10, verbose=0)
_, accuracy = model.evaluate(X_train, Y_train)
print('Accuracy: %.2f' % (accuracy*100))

dataset = pd.read_excel('surprise_test.xlsx')#################################change the file name accordingly
dataset.drop(['frame', ' confidence'], axis = 1, inplace = True)

X_test = dataset.iloc[:,:-1].to_numpy()
Y_test = dataset.iloc[:,-1].to_numpy()

predictions = (model.predict(X_test) > 0.5).astype(np.int32)
# print classification results for the first 5 test cases
for i in range(5):
	print('%s => %d (expected %d)' % (X_test[i].tolist(), predictions[i], Y_test[i]))	
_, test_accuracy = model.evaluate(X_test, Y_test)
print('Testing Accuracy: %.2f' % (test_accuracy*100))


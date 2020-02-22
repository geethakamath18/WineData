#!/usr/bin/env python
# coding: utf-8

# In[412]:


import pandas as pd
import cvxpy as cp
import numpy as np
from sklearn import metrics


# In[413]:


df=pd.read_csv("winequality-red.csv",delimiter=";")


# In[415]:


array=df.values


# In[416]:


X=array[0:1600,0:11]
Y=array[0:1600,11]


# In[417]:


X_Train=X[0:1400,0:11]
X_Test=X[1400:,0:11]
Y_Train=Y[0:1400]
Y_Test=Y[1400:]


# LINEAR REGRESSION TRAINING

# In[418]:


w=cp.Variable((11,1))
b=cp.Variable()
obj=0
for i in range(1400):
    obj+=(X_Train[i].T*w+b-Y_Train[i])**2
cp.Problem(cp.Minimize(obj), []).solve()
w=w.value
b=b.value
#print(b)


# LINEAR REGRESSION TESTING

# In[419]:


Y_Pred=np.ndarray((199,1))
for i in range(199):
    Y_Pred[i]=np.dot(X_Test[i],w)+b
print('Mean Absolute Error for Linear Regression:', metrics.mean_absolute_error(Y_Test,Y_Pred))


# HUBER LOSS TRAINING

# In[420]:


w2=cp.Variable((11,1))
b2=cp.Variable()
obj=0
for i in range(1400):
    obj+=cp.huber(X_Train[i].T*w2+b2-Y_Train[i],1)   
cp.Problem(cp.Minimize(obj),[]).solve()
w_huber=w2.value
b_huber=b2.value


# HUBER LOSS TESTING

# In[485]:


Y_Pred2=np.ndarray((199,1))
for i in range(199):
    Y_Pred2[i]=np.dot(X_Test[i],w_huber)+b_huber
print('Mean Absolute Error for Huber Loss Model:', metrics.mean_absolute_error(Y_Test,Y_Pred2))


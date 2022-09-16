#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, classification_report


# In[18]:


drug200 = pd.read_csv('https://raw.githubusercontent.com/balaji2v/Inceptez_Batch19/main/drug200.csv')


# In[19]:


drug200.head(1)


# In[20]:


drug200.head(10)


# In[21]:


drug200['Drug'] = drug200['Drug'].str.lower() 


# In[22]:


drug200


# In[28]:


drug200.sort_values(by=['Drug']).head(30)


# In[29]:


drug200.describe(include='all')


# In[45]:


drug200.loc[drug200["Cholesterol"] == "NORMAL", "Cholesterol"] = 1
drug200.loc[drug200["Cholesterol"] == "HIGH", "Cholesterol"] = 0
drug200.loc[drug200["BP"] == "HIGH", "BP"] = 2
drug200.loc[drug200["BP"] == "LOW", "BP"] = 1
drug200.loc[drug200["BP"] == "NORMAL", "BP"] = 0
drug200.loc[drug200["Sex"] == "F", "Sex"] = 0
drug200.loc[drug200["Sex"] == "M", "Sex"] = 1


# In[47]:


drug200.head(20)


# In[49]:


X = drug200.drop('Drug',1)


# In[50]:


Y = drug200.iloc[:,-1]


# In[54]:


X_Train,X_Test,Y_Train,Y_Test = train_test_split(X,Y,test_size=0.3,random_state=7)


# In[57]:


X_Train.shape,X_Test.shape,Y_Train.shape,Y_Test.shape


# In[58]:


standard_scaler = StandardScaler()


# In[60]:


X_train_scaler = standard_scaler.fit(X_Train)
X_train_trans_scaler = standard_scaler.transform(X_Train)

X_train_trans_scaler.size


# In[62]:


X_test_trans_scaler = standard_scaler.transform(X_Test)

X_test_trans_scaler.size


# In[63]:


model = KNeighborsClassifier(n_neighbors=3)


# In[64]:


model.fit(X_Train,Y_Train)


# In[65]:


y_pred = model.predict(X_Test)


# In[66]:


accuracy = accuracy_score(y_pred,Y_Test)


# In[67]:


print(accuracy)


# In[68]:


confusion_matrix(y_pred,Y_Test)


# In[69]:


print(classification_report(y_pred,Y_Test))


# In[ ]:





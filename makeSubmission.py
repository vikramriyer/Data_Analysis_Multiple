
# coding: utf-8

# Binary classification problem

# In[1]:

from sklearn.ensemble import RandomForestClassifier
from numpy import genfromtxt, savetxt


# In[5]:

df = genfromtxt(open("./dataset/bio/train.csv"), delimiter=',', dtype='f8')[1:]


# In[9]:

label = [i[0] for i in df]


# In[10]:

df_train = [i[1:] for i in df]


# In[11]:

df_test = genfromtxt(open('./dataset/bio/test.csv'), delimiter=',', dtype='f8')[1:]


# In[14]:

clf = RandomForestClassifier(n_estimators=100, n_jobs=2)


# In[15]:

clf.fit(df_train, label)


# In[29]:

predicted_probs = [[i+1, x[1]] for i, x in enumerate(clf.predict_proba(df_test))]


# In[31]:

savetxt('./submission_1.csv', predicted_probs, delimiter=',', fmt='%d,%f', header='MoleculeId,PredictedProbability', comments='')


# In[ ]:




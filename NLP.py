#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from wordcloud import WordCloud


# In[4]:


df = pd.read_csv("spam.csv",encoding='ISO-8859-1')


# In[5]:


df.head()


# In[6]:


df = df.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)


# In[7]:


df


# In[12]:


df.columns = ["labels" , "data"]


# In[13]:


df.head()


# In[20]:


def binary_map(x):
    return x.map({"ham": 0 , "spam": 1})


# In[23]:


df['b_labels'] = df['labels'].map({'ham': 0, 'spam': 1})


# In[26]:


df['b_labels'] = binary_map(df['labels'])


# In[27]:


Y = df['b_labels'].values


# In[28]:


df


# In[29]:


Y


# In[30]:


df_train, df_test, Ytrain, Ytest = train_test_split(df['data'], Y, test_size=0.33)


# In[31]:


tfidf = TfidfVectorizer(decode_error='ignore')
Xtrain = tfidf.fit_transform(df_train)
Xtest = tfidf.transform(df_test)


# In[32]:


Xtrain


# In[33]:


Xtest


# In[34]:


count_vectorizer = CountVectorizer(decode_error='ignore')
Xtrain = count_vectorizer.fit_transform(df_train)
Xtest = count_vectorizer.transform(df_test)


# In[35]:


# create the model, train it, print scores
model = MultinomialNB()
model.fit(Xtrain, Ytrain)
print("train score:", model.score(Xtrain, Ytrain))
print("test score:", model.score(Xtest, Ytest))


# In[36]:


# visualize the data
def visualize(label):
  words = ''
  for msg in df[df['labels'] == label]['data']:
    msg = msg.lower()
    words += msg + ' '
  wordcloud = WordCloud(width=600, height=400).generate(words)
  plt.imshow(wordcloud)
  plt.axis('off')
  plt.show()


# In[37]:


visualize('spam')
visualize('ham')


# In[38]:


# see what we're getting wrong
X = tfidf.transform(df['data'])
df['predictions'] = model.predict(X)


# In[39]:


# things that should be spam
sneaky_spam = df[(df['predictions'] == 0) & (df['b_labels'] == 1)]['data']
for msg in sneaky_spam:
  print(msg)


# In[40]:


# things that should not be spam
not_actually_spam = df[(df['predictions'] == 1) & (df['b_labels'] == 0)]['data']
for msg in not_actually_spam:
  print(msg)


# In[ ]:





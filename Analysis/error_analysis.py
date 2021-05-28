#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import os
import json


# In[18]:


merged = True
path = 'predictions10/'


# In[11]:


def load_json(path=path):
    """load data, use your path to the json files"""

    tweets = []
    files = []
    for f in os.listdir(path):
        if f.endswith('.json'):
            files.append(f)

    for file in files:
        with open(path+file) as f:
            for line in f:
                tweets.append(json.loads(line))
    return tweets


# In[12]:


predictions = load_json()


# In[13]:


annotations = load_json('annotation/')


# In[14]:


# create anno dict, subtract 1 from anno label to match predictions range (0-3)
anno_dict = dict()
for anno in annotations:
    anno_dict[anno['id']] = int(anno['annotation_label']) - 1


# In[15]:


# if prediction labels 2 and 3 are merged, also merge them in anno
if merged:
    for key in anno_dict.keys():
        if anno_dict[key] == 3:
            anno_dict[key] = 2


# In[16]:


# if prediction differs from annotation, keep it as error
mismatch = []
for tweet in predictions:
    if anno_dict.get(tweet['id']):
        if tweet['predict_ensemble'] != anno_dict.get(tweet['id']):
            mismatch.append([tweet['id'], tweet['date'], tweet['text'], tweet['predict_ensemble'], anno_dict.get(tweet['id'])])


# In[19]:


print('Number of errors:', len(mismatch))


# In[20]:


# create DF and save as csv
df = pd.DataFrame(mismatch)
df.columns = ['id', 'date', 'text', 'prediction', 'annotation']
df.to_csv('Analysis/error_analysis.csv', index=False)
print('File saved at Analysis/error_analysis.csv')


#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import os
import json
from collections import Counter


# In[2]:


hashtags = ['asylkrise', 'asylrecht', 'asylumseeker', 'asylumseekers', 'asylverfahren', 'austerität', 'austerity',             'debtunion', 'eurobonds', 'eurocrisis', 'eurokrise', 'eusolidarität', 'eusolidarity', 'exiteu',             'fiscalunion', 'fiskalunion', 'flüchtling', 'flüchtlinge', 'flüchtlingskrise', 'flüchtlingswelle',             'leavenoonebehind', 'migrationskrise', 'niewieder2015', 'noasyl', 'opentheborders', 'refugee',             'refugeecrisis', 'refugees', 'refugeesnotwelcome', 'refugeeswelcome', 'remigration', 'rightofasylum',             'schuldenunion', 'seenotrettung', 'standwithrefugees', 'transferunion', 'wirhabenplatz', 'wirschaffendas',             'fluechtling', 'fluechtlinge', 'fluechtlingswelle', 'eusolidaritaet', 'austeritaet', 'fluechtlingskrise']

eurocrisis_tags = ['austerität', 'austerity', 'debtunion', 'eurobonds', 'eurocrisis', 'eurokrise', 'eusolidarität',                    'eusolidarity', 'exiteu', 'fiscalunion', 'fiskalunion', 'schuldenunion', 'transferunion',                    'eusolidaritaet', 'austeritaet']

refugee_tags = ['asylkrise', 'asylrecht', 'asylumseeker', 'asylumseekers', 'asylverfahren', 'flüchtling',                 'flüchtlinge', 'flüchtlingskrise', 'flüchtlingswelle', 'leavenoonebehind', 'migrationskrise',                 'niewieder2015', 'noasyl', 'opentheborders', 'refugee', 'refugeecrisis', 'refugees',                 'refugeesnotwelcome', 'refugeeswelcome', 'remigration', 'rightofasylum', 'seenotrettung',                 'standwithrefugees', 'wirhabenplatz', 'wirschaffendas', 'fluechtling', 'fluechtlinge',                 'fluechtlingswelle', 'fluechtlingskrise']


# In[3]:


def load_json_in_df(path='predictions10/'):
    """load data, use your path to the json files."""
    tweets = []
    files = []
    for f in os.listdir(path):
        if f.endswith('.json'):
            files.append(f)

    for file in files:
        with open(path + file) as f:
            for line in f:
                tweets.append(json.loads(line))
    print('Number of Tweets: ', len(tweets))
    
    # convert important parts to DataFrame
    abstract = [[tweet['id'],tweet['date'],tweet['text'],tweet['predict_ensemble'],tweet['hashtags']] for tweet in tweets]
    df_abstract = pd.DataFrame(abstract)
    df_abstract.columns = ['id','date','text', 'prediction', 'hashtags']   
    df_abstract['date'] = pd.to_datetime(df_abstract['date'])
    df_abstract['date'] = pd.to_datetime(df_abstract['date'].apply(lambda x: "%d-%d-%d" % (x.year, x.month, x.day)))
    df_abstract.set_index('date', inplace=True)
    return df_abstract


# In[4]:


df = load_json_in_df()


# In[7]:


def create_daily_statistic(df_abstract, filename='Analysis/stats_per_day.csv', start="2019-09-01", end="2020-08-31", filter_hashtags=None):
    """creates a csv and DataFrame containing the # of Solidary, Anti-Solidary, Other and the Top 10 most common
    hashtags per day and returns the DF"""
    
    if filter_hashtags:
        df_abstract = df_abstract[df_abstract['hashtags'].apply(lambda x: bool(set(filter_hashtags).intersection(x)))]
    daterange = pd.date_range(start=start,end=end).tolist()
    stats_per_day = []
    for date in daterange:
        time = date.strftime("%Y-%m-%d")
        tmp = df_abstract[time]
        tags = tmp['hashtags'].tolist()
        tags = [item for sublist in tags for item in sublist]
        c = Counter(tags)
        top10 = ''
        for tag, freq in c.most_common(10):
            top10 += str(tag) + ': ' + str(freq) + '\t'
        stats_per_day.append([date, len(tmp[tmp['prediction']==0]), len(tmp[tmp['prediction']==1]), len(tmp[tmp['prediction']==2]), top10])
        c.clear()
        
    df = pd.DataFrame(stats_per_day)
    df.columns = ['Date', 'Solidary', 'Anti-Solidary', 'Other', '10 most common hashtags']
    df.to_csv(filename, index=False)
    print('Daily Stats created at', filename)
    return df


# In[8]:


stats_df = create_daily_statistic(df)


# In[9]:


eurocrisis_stats_df = create_daily_statistic(df, filename='Analysis/eurocrisis_stats_per_day.csv', filter_hashtags=eurocrisis_tags)


# In[10]:


refugeecrisis_stats_df = create_daily_statistic(df, filename='Analysis/refugeecrisis_stats_per_day.csv', filter_hashtags=refugee_tags)


# In[16]:


def create_timeframe_sample(df_abstract, start, end, folder='Analysis/', filter_hashtags=None, number_samples=20):
    """Creates a csv and DataFrame with a number of samples per day in the specified timeframe and returns the DF."""
    if filter_hashtags:
        df_abstract = df_abstract[df_abstract['hashtags'].apply(lambda x: bool(set(filter_hashtags).intersection(x)))]
    daterange = pd.date_range(start=start,end=end).tolist()
    # create samples per day
    samples_per_day = []
    for date in daterange:
        time = date.strftime("%Y-%m-%d")
        tmp = df_abstract[time]
        t = tmp.sample(number_samples)
        samples_per_day.append(t)
    samples_per_day = pd.concat(samples_per_day)
    samples_per_day.to_csv(folder + start + 'to_' + end + '_daily_samples.csv', index=True)
    print('Samples created at', folder + start + 'to_' + end + '_daily_samples.csv')
    return samples_per_day


# In[17]:


sample = create_timeframe_sample(df, '2020-07-17', '2020-07-30', folder = 'Analysis/eurocrisis_', filter_hashtags=eurocrisis_tags, number_samples=15)


# In[18]:


sample = create_timeframe_sample(df, '2020-08-11', '2020-08-12', folder = 'Analysis/eurocrisis_', filter_hashtags=eurocrisis_tags, number_samples=15)


# In[19]:


sample = create_timeframe_sample(df, '2020-02-28', '2020-03-18', folder = 'Analysis/refugeecrisis_', filter_hashtags=refugee_tags, number_samples=20)


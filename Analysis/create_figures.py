#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import os
import json
from collections import Counter
import matplotlib.pyplot as plt
import re


# In[2]:

# list of all hashtags used. additions to handle umlaut. adjust if necessary
hashtags = ['asylkrise', 'asylrecht', 'asylumseeker', 'asylumseekers', 'asylverfahren', 'austerität', 'austerity',             'debtunion', 'eurobonds', 'eurocrisis', 'eurokrise', 'eusolidarität', 'eusolidarity', 'exiteu',             'fiscalunion', 'fiskalunion', 'flüchtling', 'flüchtlinge', 'flüchtlingskrise', 'flüchtlingswelle',             'leavenoonebehind', 'migrationskrise', 'niewieder2015', 'noasyl', 'opentheborders', 'refugee',             'refugeecrisis', 'refugees', 'refugeesnotwelcome', 'refugeeswelcome', 'remigration', 'rightofasylum',             'schuldenunion', 'seenotrettung', 'standwithrefugees', 'transferunion', 'wirhabenplatz', 'wirschaffendas']
additions = ['fluechtling', 'fluechtlinge', 'fluechtlingswelle', 'eusolidaritaet',              'austeritaet', 'fluechtlingskrise']


# In[3]:


def load_json(path='predictions10/'):
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


# In[4]:


def create_dict(tweets, hashtag_field=None):
    """ Converts json data to a dict of hashtags as keys and a list of predictions containing that hashtag as values."""
    pred_dict = dict()
    # when using self defined json hashtag field
    if hashtag_field:
        for tag in hashtags:
            tmp = []
            for tweet in tweets:
                if tag in tweet[hashtag_field]:
                    tmp.append(tweet)
            pred_dict[tag] = tmp
        
        # handle ä/ae,ö/oe,ü/ue
        for tag in additions:
            tmp = []
            for tweet in tweets:
                if tag in tweet[hashtag_field]:
                    tmp.append(tweet)
            pred_dict[tag] = tmp
    # when using official twitter json
    else:
        for tag in hashtags:
            tmp = []
            for tweet in tweets:
                tweet_tags = [t['text'].lower() for t in tweet['entities']['hashtags']]
                if tag in tweet_tags:
                    tmp.append(tweet)
            pred_dict[tag] = tmp
        
        # handle ä/ae,ö/oe,ü/ue
        for tag in additions:
            tmp = []
            for tweet in tweets:
                tweet_tags = [t['text'].lower() for t in tweet['entities']['hashtags']]
                if tag in tweet_tags:
                    tmp.append(tweet)
            pred_dict[tag] = tmp
    
    # combine with corresponding tag. adjust if list of hashtags with umlaut changes
    pred_dict['flüchtling'] = pred_dict['flüchtling'] + pred_dict['fluechtling']
    pred_dict['flüchtlinge'] = pred_dict['flüchtlinge'] + pred_dict['fluechtlinge']
    pred_dict['flüchtlingswelle'] = pred_dict['flüchtlingswelle'] + pred_dict['fluechtlingswelle']
    pred_dict['eusolidarität'] = pred_dict['eusolidarität'] + pred_dict['eusolidaritaet']
    pred_dict['austerität'] = pred_dict['austerität'] + pred_dict['austeritaet']
    pred_dict['flüchtlingskrise'] = pred_dict['flüchtlingskrise'] + pred_dict['fluechtlingskrise']
    # delete leftover temporary keys
    for tag in additions:
        del pred_dict[tag]
    
    return pred_dict


# In[5]:


predictions = load_json()


# In[6]:


pred_dict = create_dict(predictions, 'hashtags')


# In[7]:


annotations = load_json('annotation/')


# In[8]:


anno_dict = create_dict(annotations)


# In[9]:


def create_figures(pred_dict, pred_field='predict_ensemble', folder='Analysis/Pred_Figures/', title='Ensemble Predictions', merged=True):
    """Creates figures showing the label distribution per hashtag."""
    for key,values in pred_dict.items():
        predictions = [int(tweet[pred_field]) for tweet in values]
        c = Counter(predictions)
        if merged:
            labels = ['solidary',  'anti-\nsolidary', 'other']
            colors = ['blue', 'red', 'yellow']
            bars = plt.bar(labels, [c[0], c[1], c[2]], color=colors)
            plt.title(title+ ':{} Data Distribution'.format(key))
        
            # Add counts above the bar graph
            for rect in bars:
                height = rect.get_height()
                plt.text(rect.get_x() + rect.get_width()/2.0, height, '%d' % int(height), ha='center', va='bottom')
        
        else:
            labels = ['solidary',  'anti-\nsolidary', 'ambivalent', 'non-\napplicable']
            colors = ['blue', 'red', 'yellow', 'orange']
            bars = plt.bar(labels, [c[1], c[2], c[3], c[4]], color=colors)
            plt.title(title+ ':{} Data Distribution'.format(key))
        
            # Add counts above the bar graph
            for rect in bars:
                height = rect.get_height()
                plt.text(rect.get_x() + rect.get_width()/2.0, height, '%d' % int(height), ha='center', va='bottom')
        
        plt.savefig(folder+ '{}'.format(key))
        plt.close()
        c.clear()
    print('Figures created at '+ folder)


# In[10]:


create_figures(pred_dict)


# In[11]:


create_figures(anno_dict, 'annotation_label', 'Analysis/Anno_Figures/', 'Annotations', False)


# In[12]:


def create_comparison_figures(pred_dict, anno_dict, pred_field='predict_ensemble', anno_field='annotation_label', folder='Analysis/Comparison_Figures/'):
    """Creates Figures that compare the percentage distribution of the labels between the predicitons and annotations per hashtag."""
    for key,values in pred_dict.items():
        p_sents = [tweet[pred_field] for tweet in values]
        p_c = Counter(p_sents)
        p_percents = [p_c[0]/(p_c[0]+p_c[1]+p_c[2]), p_c[1]/(p_c[0]+p_c[1]+p_c[2]), p_c[2]/(p_c[0]+p_c[1]+p_c[2])]
    
        a_sents = [tweet[anno_field] for tweet in anno_dict[key]]
        a_c = Counter(a_sents)
        a_percents = [a_c['1']/(a_c['1']+a_c['2']+a_c['3']+a_c['4']), a_c['2']/(a_c['1']+a_c['2']+a_c['3']+a_c['4']), (a_c['3']+a_c['4'])/(a_c['1']+a_c['2']+a_c['3']+a_c['4'])]
    
        labels = ["pred-\nsolidary", "anno-\nsolidary",  "pred-\nanti-\nsolidary", "anno-\nanti-\nsolidary",  "pred-\nother", "anno-\nother"]
        colors = ['blue', 'cornflowerblue', 'red', 'lightcoral', 'yellow', 'lemonchiffon']
        percents = [p_percents[0], a_percents[0], p_percents[1], a_percents[1], p_percents[2], a_percents[2]]
        percents = [100*a for a in percents]
        bars = plt.bar(labels, percents, color=colors)
        plt.title("Ensemble Predictions vs. Annotations:\n{} % Data Distribution".format(key))
    
        # Add counts above the bar graph
        for rect in bars:
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width()/2.0, height, '%.1f%%' % height, ha='center', va='bottom')
                
        plt.savefig(folder+key)
        plt.close()
        p_c.clear()
        a_c.clear()
    print('Comparison Figures created at '+ folder)


# In[14]:


create_comparison_figures(pred_dict, anno_dict)


# In[15]:


def create_all_data_stats(json, pred_field='predict_ensemble', title='All Model Predictions Data Distribution', path='Analysis/predictions_all_data', merged=True):
    """Creates a bar and pie chart showing the label distribution over the whole dataset."""
    predictions = [int(tweet[pred_field]) for tweet in json]
    if merged:
        c = Counter(predictions)
        labels = ["solidary", "anti-\nsolidary", "other"]
        colors = ['blue', 'red', 'yellow']
        freqs = [c[0], c[1], c[2]]
    else:
        c = Counter(predictions)
        labels = ["solidary", "anti-\nsolidary", "ambivalent", "non-\napplicable"]
        colors = ['blue', 'red', 'yellow', 'orange']
        freqs = [c[1], c[2], c[3], c[4]]

    pie = plt.pie(freqs, colors=colors, labels=labels, autopct='%.1f%%', startangle=90)
    plt.axis("image")
    plt.title(title)
    plt.savefig(path+'_pie')
    plt.close()
    
    bars = plt.bar(labels,freqs, color=colors)
    plt.title(title)
        
    # Add counts above the bar graph
    for rect in bars:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2.0, height, '%d' % int(height), ha='center', va='bottom')
    plt.savefig(path+'_bars')
    plt.close()
    c.clear()
    print('Whole Data Charts created at '+ path)


# In[16]:


create_all_data_stats(predictions)


# In[17]:


create_all_data_stats(annotations, 'annotation_label', 'Annotation Data Distribution', 'Analysis/annotations_all_data', False)


# In[ ]:





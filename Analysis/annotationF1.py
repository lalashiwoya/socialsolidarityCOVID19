#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn import metrics


# In[2]:


path_to_directory = 'annotation/'
all_data = False


# In[8]:


def load_data_DT(path= path_to_directory, all_data=False):
    """Loads the annotation data. 
    Args:
        path (str): path to the directory containing the .csv files
        all_data (bool): if False only the annotations between Tobias and Daniel are loaded. If True the earlier
                         ones with different annotators are also loaded and treated as annotated by Tobias and Daniel
    Returns: DataFrame with the annotations as well as the classes 3 and 4 merged"""
    a1 = pd.read_csv(path + 'annotation301-339.csv', sep=';')
    a1 = a1[:39]
    a1 = a1.drop(columns=['Ying', 'Dan'])
    
    a2 = pd.read_csv(path + 'annotation5.csv')
    a2 = a2.drop(columns=['Unnamed: 4', 'Unnamed: 6', 'Comment Alexandra'])
    
    a3 = pd.read_csv(path + 'annotation6a.csv')
    a3 = a3.drop(columns=['Unnamed: 4', 'Unnamed: 6'])
    
    a4 = pd.read_csv(path + 'annotation6b.csv')
    a4 = a4.drop(columns=['Unnamed: 4', 'Unnamed: 6'])
    
    if all_data:
        d_y = pd.read_csv(path + 'annotation_1-150.csv', sep=';')
        d_y = d_y.drop(columns=['Dan', 'Tobias', 'Agreement.1', 'Comment Alexandra'])
        d_y.columns=['id', 'date', 'text', 'Tobias', 'Daniel', 'Agreement']
        d_y = d_y[:150]
        
        d_t = pd.read_csv(path + 'annotation151-300.csv', sep=';')
        d_t = d_t.drop(columns=['Daniel', 'Ying', 'Agreement.1'])
        d_t.columns=['id', 'date', 'text', 'Daniel', 'Tobias', 'Agreement']
        d_t = d_t[:150]
        
        anno = pd.concat([d_y, d_t, a1,a2,a3,a4], sort=False)
    
    else:
        anno = pd.concat([a1,a2,a3,a4], sort=False)
    # convert float to string to compare with string annotations
    anno['Daniel'] = anno['Daniel'].apply(lambda x: str(int(x)) if (type(x) != str) else x)
    anno['Tobias'] = anno['Tobias'].apply(lambda x: str(int(x)) if (type(x) != str) else x)
    
    #create colums for the merged label 3 and 4
    anno['Merged_Daniel'] = anno['Daniel']
    anno.loc[(anno.Merged_Daniel == '4'), 'Merged_Daniel'] = '3'
    anno['Merged_Tobias'] = anno['Tobias']
    anno.loc[(anno.Merged_Tobias == '4'), 'Merged_Tobias'] = '3'
    anno['Merged_Agreement'] = anno['Agreement']
    anno.loc[(anno.Merged_Agreement == '4'), 'Merged_Agreement'] = '3'
    return anno


# In[4]:


def calc_f1s(anno, all_data=False):
    """Calculates the F1 Score between the annotators and the agreement and themselves. Prints out the results.
    Args:
        anno (DataFrame): DataFrame containing the data.
        all_data (bool): If False data is supposed to contain only the annotation between Tobias and Daniel. If True
                         data over different annotators"""
    f1_d = metrics.f1_score(anno['Agreement'], anno['Daniel'], labels=['1','2','3','4'], average='macro')
    f1_t = metrics.f1_score(anno['Agreement'], anno['Tobias'], labels=['1','2','3','4'], average='macro')
    f1_vs = metrics.f1_score(anno['Daniel'], anno['Tobias'], labels=['1','2','3','4'], average='macro')
    f1_d_m = metrics.f1_score(anno['Merged_Agreement'], anno['Merged_Daniel'], labels=['1','2','3'], average='macro')
    f1_t_m = metrics.f1_score(anno['Merged_Agreement'], anno['Merged_Tobias'], labels=['1','2','3'], average='macro')
    f1_vs_m = metrics.f1_score(anno['Merged_Daniel'], anno['Merged_Tobias'], labels=['1','2','3'], average='macro')
    
    if all_data:
        print('Combined the 150 Tweets of Dan with the rest of Daniel into Daniel and 150 Tweets of Ying with the rest of Tobias into Tobias. So the first 300 Tweets were across different annotators but respresent our whole Dataset.')
    print('Macro F1 Scores, accounting for ', len(anno), ' Tweets:')
    print('Macro F1 Daniel vs. Agreement:\t\t\t', f1_d*100)
    print('Macro F1 Tobias vs. Agreement:\t\t\t', f1_t*100)
    print('Macro F1 Daniel vs. Tobias:\t\t\t', f1_vs*100)
    print('Macro F1 Merged Classes Daniel vs. Agreement:\t', f1_d_m*100)
    print('Macro F1 Merged Classes Tobias vs. Agreement:\t', f1_t_m*100)
    print('Macro F1 Merged Classes Daniel vs. Tobias:\t', f1_vs_m*100)


# In[5]:


def calc_cohens_kappa(anno, all_data=False):
    """Calculates the F1 Score between the annotators and the agreement and themselves. Prints out the results.
    Args:
        anno (DataFrame): DataFrame containing the data.
        all_data (bool): If False data is supposed to contain only the annotation between Tobias and Daniel. If True
                         data over different annotators"""
    kappa = metrics.cohen_kappa_score(anno['Daniel'], anno['Tobias'], labels=['1','2','3','4'])
    merged_kappa = metrics.cohen_kappa_score(anno['Merged_Daniel'], anno['Merged_Tobias'], labels=['1','2','3'])
    if all_data:
        print('Combined the 150 Tweets of Dan with the rest of Daniel into Daniel and 150 Tweets of Ying with the rest of Tobias into Tobias. So the first 300 Tweets were across different annotators but respresent our whole Dataset.')
    print('Cohens Kappa between Daniel and Tobias, accounting for ', len(anno), ' Tweets')
    print('4 Classes:\t\t', kappa)
    print('Merged last 2 classes: \t', merged_kappa)


# In[6]:


anno = load_data_DT(all_data=all_data)
calc_f1s(anno, all_data)
print()
calc_cohens_kappa(anno,all_data)


# In[9]:


all_data = True
anno = load_data_DT(all_data=all_data)
calc_f1s(anno, all_data)
print()
calc_cohens_kappa(anno,all_data)


# In[ ]:





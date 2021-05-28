#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import os
import json
import geopandas as gpd
from shapely.geometry import Point, Polygon
import folium


# In[2]:


def load_json(path='twitter_data/json/'):
    """load data, use your path to the original crawled Twitter json files."""

    tweets = []
    files = []
    for f in os.listdir(path):
        if f.endswith('.json'):
            files.append(f)

    for file in files:
        with open(path+file) as f:
            for line in f:
                tweets.append(json.loads(line))
    print(len(tweets), ' Tweets loaded.')
    return tweets


# In[3]:


def extract_geodata(tweets, map_file='twitter_locations_from_google.json'):
    """Extracts all available tweets with geodata associated. Profile locations are mapped with the provided mapping file if possible. Returns a list."""
    # load google location mapping data

    with open(map_file) as f:
        mapped_loc = json.load(f)
    
    #extract different geo-information of tweets. exclude in less informative if more precise geo-information already available

    coords = [[tweet['id'], tweet['coordinates']['coordinates'][0], tweet['coordinates']['coordinates'][1],               tweet['full_text']] for tweet in tweets if tweet['coordinates']]
    print(len(coords), ' Tweets with exact coordinates')

    # use center point of bounding box
    place = [[tweet['id'],               (float(tweet['place']['bounding_box']['coordinates'][0][0][0])              + float(tweet['place']['bounding_box']['coordinates'][0][1][0])) / 2,                (float(tweet['place']['bounding_box']['coordinates'][0][0][1])              + float(tweet['place']['bounding_box']['coordinates'][0][3][1])) / 2,                tweet['full_text']] for tweet in tweets if tweet['place'] and not tweet['coordinates']]
    print(len(place), ' additionally Tweets with a place defined by a bounding box.')

    loc = [[tweet['id'], tweet['user']['location'], tweet['full_text']] for tweet in tweets if tweet['user']['location'] and not tweet['coordinates'] and not tweet['place']]
    print(len(loc), ' additionally Tweets with a user profile location.')
    
    # replace location string with google maps api geodata if available
    coords_extended_loc = []

    for l in loc:
        if l[1] in mapped_loc and mapped_loc[l[1]]:
            coords_extended_loc.append([l[0], mapped_loc[l[1]]['geometry']['location']['lng'], mapped_loc[l[1]]['geometry']['location']['lat'], l[2]])
    print(len(coords_extended_loc), ' Tweets with profile locations sucessfully mapped to coordinates.')

    all_geodata = coords + place + coords_extended_loc
    print(len(all_geodata), ' Tweets in total with associated coordinates.')
    return all_geodata


# In[4]:


def geofiltering_eu(all_geodata):
    """Filters by a bounding box for EU and returns the filtered list."""
    # filter by a big rough bounding box 
    filterted_geodata = [item for item in all_geodata if (float(item[1]) > -13.1 and float(item[1]) < 41.5 and float(item[2]) > 35.5 and float(item[2]) < 72.1)]
    print(len(filterted_geodata), ' Tweets with EU geotag.')
    return filterted_geodata


# In[13]:


def geofiltering_ger(all_geodata):
    """Precise geofiltering for Germany, returns filtered list"""
    # load gpd shape of Germany
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    ger =  world[world.name == "Germany"]
    
    # filter by coords in Germany
    filtered_geodata = [data for data in all_geodata if Point(data[1], data[2]).within(ger['geometry'][0])]
    print(len(filtered_geodata), ' Tweets with German geotag.')
    return filtered_geodata


# In[21]:


def save_dict(geodata, file):
    """Saves filtered data as a json dict."""
    geo_dict = dict()
    for g in geodata:
        geo_dict[g[0]] = g[3]
        
    with open(file, 'w') as f:
        json.dump(geo_dict, f)
    print('Saved at', file)


# In[34]:


def visualize_on_map(geodata, file):
    """Visualizes the (filtered) data on a map and saves that as html file"""
    # plausability check by visualizing with folium
    def plotDots(dataframe):
    # reading geodata into folium map
        folium.CircleMarker(location=[dataframe.lat,dataframe.long],
                        radius=6,
                        color = "red",
                        fill=True,
                        fill_color='red').add_to(twitter_map)
    #create df
    df = gpd.GeoDataFrame(geodata)
    df.columns = ['id', 'long', 'lat', 'text']
    
    # create folium map 
    twitter_map = folium.Map(prefer_canvas=True)


    # Apply plotDot to dataframe
    df.apply(plotDots, axis = 1)

    # zoom in
    twitter_map.fit_bounds(twitter_map.get_bounds())

    # save map
    twitter_map.save(file)
    print('Saved Map in', file)


# In[6]:


tweets = load_json()


# In[7]:


all_geodata = extract_geodata(tweets)


# In[8]:


eu = geofiltering_eu(all_geodata)


# In[14]:


ger = geofiltering_ger(all_geodata)


# In[22]:


save_dict(eu, 'eu_dict.json')
save_dict(ger, 'germany_dict.json')


# In[35]:


visualize_on_map(ger, 'Analysis/filtered_germany_map.html')


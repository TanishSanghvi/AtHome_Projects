#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 14 18:42:07 2022

@author: apple
"""

import pandas as pd
from sklearn.neighbors import NearestNeighbors


df_ratings = pd.read_csv('/Users/apple/Downloads/take_home_ss_ratings.csv')

#Converting 0 values to -1 as items for which users have not given a rating will be mapped to 0
df_ratings['rating'].replace({0:-1}, inplace = True) 

df_clean = df_ratings[['user_id', 'item_id', 'rating']]

#Converting data set to a pivot table with items as rows and users as columns
df = pd.pivot_table(df_clean, values='rating', index=['item_id'],
                    columns=['user_id'], fill_value=0) 

#Determining cosine similarity between items. Looking at 75 closest neighbours as data is very sparse
knn = NearestNeighbors(metric='cosine', algorithm='auto')
knn.fit(df.values)
distances, indices = knn.kneighbors(df.values, n_neighbors=75)

#Function that returns a rating given a user and an item
def predict_rating(df, u, i):
    
    item = i
    user = u
    
    #Determining most similar items and their distances
    sim_items = list(indices[item])[1:] #The most similar item will always be the item itself
    sim_distances = list(1-distances[item])[1:] #Subtracting distance from 1 to get dimilarity
    
    sum_distances = sum(sim_distances)
    
    print('Nearest items to item '+str(item)+' are:', sim_items, '\n')
    
    sim_distances_copy = []
    numerator = 0
    
    #Looping through similar items to get predicted rating
    for s in range(0, len(sim_distances)):
        
        #Using only similar items which the user has rated
        if df.iloc[sim_items[s], user] != 0:
            
            numerator = numerator + sim_distances[s] * df.iloc[sim_items[s], user]
            sim_distances_copy.append(sim_items[s])
            
    if len(sim_distances_copy) > 0:
        
        predicted_r = numerator/sum_distances
        
        if predicted_r > 0:
            print('The rating for item '+ str(i) +' by user '+ str(u) +' is: 1')
        else:
            print('The rating for item '+ str(i) +' by user '+ str(u) +' is: 0')
        
    else:
        print('Not enough data to determine significantly')
        
predict_rating(df, 5, 15)










    
    
    
    
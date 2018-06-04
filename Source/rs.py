#!/usr/bin/env python
"""
    File name: 
    Description:
    Author: Rishabh Gupta
    Date created:
    Date last modified:
    Python Version: 2.7
"""

import pandas as pd
ROOT = "D:/Office/Project/RecommendationSystem/Source/"
DATA_ROOT = ROOT + "Data/ml-100k/"

u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']

user_data = pd.read_csv(DATA_ROOT + "/u.user", sep='|', names=u_cols, encoding='latin-1')

# Actual data
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings_data = pd.read_csv(DATA_ROOT + '/u.data', sep='\t', names=r_cols,
 encoding='latin-1')

#Reading items file:
i_cols = ['movie id', 'movie title' ,'release date','video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure',
 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
items = pd.read_csv(DATA_ROOT + '/u.item', sep='|', names=i_cols,
 encoding='latin-1')

# ratings_base = pd.read_csv(DATA_ROOT + '/ua.base', sep='\t', names=r_cols, encoding='latin-1')
# ratings_test = pd.read_csv(DATA_ROOT + '/ua.test', sep='\t', names=r_cols, encoding='latin-1')

# print ratings_base.shape, ratings_test.shape

# import graphlab

# https://cambridgespark.com/content/tutorials/implementing-your-own-recommender-systems-in-Python/index.html
# from sklearn import cross_validation as cv
from sklearn.model_selection import train_test_split

train_data, test_data = train_test_split(ratings_data, test_size=0.25)
print "hi"

# 100000, 4
# 75000, 4
# 25000, 4
# Create user-item matrix
# no of users * no of  items[movies]

# movie movie
# rating
import numpy as np
training_no_movies = len(ratings_data.movie_id.unique())
training_no_users = len(ratings_data.user_id.unique())
train_data_matrix = np.zeros(shape=[training_no_users,training_no_movies])
print train_data_matrix.shape
for line in train_data.itertuples():
    # line[1] = user index
    # line[2] = movie index
    # line[3] = rating
    train_data_matrix[line[1]-1, line[2]-1] = line[3]

test_data_matrix = np.zeros((training_no_users, training_no_movies))
for line in test_data.itertuples():
    test_data_matrix[line[1]-1, line[2]-1] = line[3]

print "hi"
from sklearn.metrics.pairwise import pairwise_distances

user_similarity = pairwise_distances(train_data_matrix, metric='cosine')
item_similarity = pairwise_distances(train_data_matrix.T, metric='cosine')
print "hi"
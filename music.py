import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

data = pd.read_csv('music_data.csv')

train_data, test_data = train_test_split(data, test_size=0.2)

train_matrix = train_data.pivot_table(index='userid', columns='artistid', values='plays').fillna(0)

user_similarity = cosine_similarity(train_matrix)
item_similarity = cosine_similarity(train_matrix.T)

def predict(plays, similarity, type='user'):
    if type == 'user':
        mean_user_rating = plays.mean(axis=1)
        ratings_diff = (plays - mean_user_rating[:, np.newaxis])
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
        pred = plays.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    return pred

user_prediction = predict(train_matrix.values, user_similarity, type='user')
item_prediction = predict(train_matrix.values, item_similarity, type='item')

def recommend_items(userid, prediction_matrix, n=5):
    user_row = userid - 1
    sorted_indices = np.argsort(prediction_matrix[user_row])[::-1]
    recommended_items = []
    for i in range(n):
        recommended_items.append(sorted_indices[i])
    return recommended_items

userid = 1
recommended_items = recommend_items(userid, user_prediction)
print("Recommended items for user", userid, ":", recommended_items)

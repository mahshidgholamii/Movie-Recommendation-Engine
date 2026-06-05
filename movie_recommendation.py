"""
Building a Movie Recommendation Engine

this project builds a simple recommendation that suggest movies based on
similarity of their content features like: keywords, genres,
cast, and director.

Built with Python, using Pandas for data processing and Scikit-learn for similarity computation.

Author: Mahshid Gholami

"""


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.metrics.pairwise import cosine_similarity


def get_title_from_index(index):
    return df[df.index == index]["title"].values[0]

def get_index_from_title(title):
    return df[df.title == title]["index"].values[0]


# 1. Load movie dataset using pandas
df = pd.read_csv("movie_dataset.csv")

print(df.head())
print(df.columns)

# 2. Select relevant features (keywords, genres, cast, director)
features = ['keywords', 'genres', 'cast', 'director']

# 3. Fill missing values to avoid processing errors
for features in features:
    df[features] = df[features].fillna('')

# 4. Combine selected features into a single string per movie  
def features_combination(row):
    try:
        return row['keywords'] +" "+row["cast"] +" "+row["genres"] +" "+row["director"]

    except Exception as e:
        print("Error: ", row)

df["feature_combination"] = df.apply(features_combination, axis=1)
print(f"the combination of features:\n\n{df["feature_combination"].head()}")

# 5. Convert text data into a count matrix using CountVectorizer
cv = CountVectorizer()
count_matrix = cv.fit_transform(df["feature_combination"])

# 6. Compute cosine similarity between all movies
cosign_similarity = cosine_similarity(count_matrix)
user_favorite_movie = "Batman v Superman: Dawn of Justice"

movie_index = get_index_from_title(user_favorite_movie)

the_same_movies = list(enumerate(cosign_similarity[movie_index]))
sorted_the_same_movies = sorted(the_same_movies,key=lambda x:x[1], reverse=True)

# 7. Recommend top similar movies based on a chosen favorite movie
i = 0
print("\nRecommended Movies: ")
while i < len(sorted_the_same_movies) and i < 50:
    movies = sorted_the_same_movies[i]
    print(get_title_from_index(movies[0]))
    i += 1

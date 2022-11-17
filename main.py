import pandas as pd
import numpy as np

df = pd.read_csv("NetflixDataset.csv")
#dropping nan values from the three rows
df.dropna(subset=['Director', 'Genre', 'Actors'], inplace=True)
df.drop_duplicates(subset="Title", keep="first", inplace=True)
movies = df
#dropping duplicates
#resetting the index
movies = movies.reset_index()
movies = movies.drop(['index'],axis=1)
features = []
for i in range(len(movies)):
    features.append(movies['Title'][i]+', ('+movies['Director'][i]+'), ('+movies['Actors'][i]+'), '+movies['Genre'][i])

#adding the features in a new column
movies['combined features'] = features
#print(movies.iloc[0, :])

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

vectors = CountVectorizer().fit_transform(movies['combined features'])
similarity_matrix = cosine_similarity(vectors)
#print(similarity_matrix)

def recommend(Title, number):
    mindex = 0
    k = 0
    for i in range(len(movies)):
        if Title.lower() == movies['Title'][i].lower():
            mindex = i
            k = 1
            break
    if k == 1:
        result = similarity_matrix[mindex]
        result = list(np.argsort(result))
        result.reverse()
        recommend_movies = []
        for i in range(1,number+1):
            recommend_movies.append(movies['Title'][result[i]])
        return recommend_movies
    return "Movie not found"

name = input("Movie name: ")
n = 10
print(f"{n} movies that you might like 👇")
print()
[print(i) for i in recommend(name, n)]

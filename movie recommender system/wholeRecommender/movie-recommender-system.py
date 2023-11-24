import ast
import numpy as np 
import pandas as pd
movies = pd.read_csv(r'C:\Users\91934\Desktop\ML project\movie recommender system\wholeRecommender\tmdb_5000_movies.csv')
credits =  pd.read_csv(r'C:\Users\91934\Desktop\ML project\movie recommender system\wholeRecommender\tmdb_5000_credits.csv')

movies = movies.merge(credits, on='title')
movies.head(1)

# genres
#id
#keyword
#title
#overviews
#cast
#crew
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
# movies.info()

# print(movies)

movies.isnull().sum()
movies.dropna(inplace=True)
movies.duplicated().sum()
movies.iloc[0].genres
# print(movies)

# [{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]  
# ['Action', 'Adventure', 'Fantasy', 'SciFi']
def convert(obj):
    L = []
    for i in ast.literal_eval(obj): # use to convert the string into list form
        L.append(i['name'])
    return L
movies['genres'] = movies['genres'].apply(convert) # all the genres are right place now
movies['keywords'] = movies['keywords'].apply(convert)
def convert3(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):    # For each movie we are getting three actor name 
        if counter != 3 :
            L.append(i['name'])
            counter += 1
        else: 
            break
    return L
movies['cast'] = movies['cast'].apply(convert3)
def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i ['job'] == 'Director' : 
            L.append(i['name'])
            break
    return L 
movies['crew'] = movies['crew'].apply(fetch_director)
# print(movies.head)
# print(movies['overview'][0]) # This overview is in the for of string so we'll convert it into list.
movies['overview'] = movies['overview'].apply(lambda x:x.split())
# Now all the things in the form of list. Now we will concadinate all these 
movies['genres'] = movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])  # we need to remove space for confusion
movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast'] = movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew'] = movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])
# We concadinate all the data
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
# Now all the data combines as tags so we can use only tags.
new_df = movies[['movie_id', 'title', 'tags']]
new_df['tags'] = new_df['tags'].apply(lambda x:" ".join(x))
new_df['tags'] = new_df['tags'].apply(lambda x:x.lower())
# print(new_df['tags'][0])

# we are converting movies into text vectors and using "bag of wrods" technique.
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()
# We are using nltk for stemming (makeing {loving, loved, love} to {love, love, love})
import nltk
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

def stem(text):
    y = []

    for i in text.split():
        y.append(ps.stem(i))    # converted into list

    return " ".join(y)   # Again converting into string.

new_df['tags'] = new_df['tags'].apply(stem)

from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(vectors)

sorted(list(enumerate(similarity[0])), reverse=True, key=lambda x:x [1])[1:6]

def recommend(movie): 
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x:x[1])[1:6]

    for i in movies_list:
        print(new_df.iloc[i[0]].title)

print(recommend('Avatar'))

import pickle 
pickle.dump(new_df, open('movies.pkl', 'wb'))

pickle.dump(new_df.to_dict(), open('movie_dict.pkl', 'wb'))

pickle.dump(new_df.to_dict(), open('similarity.pkl', 'wb'))




cv.get_feature_names_out()
    
# print(ps.stem('loved'))



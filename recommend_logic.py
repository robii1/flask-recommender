import pandas as pd
import os
import json
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process  

#  filsti for "movies.csv" ( lokalt og på Render)
csv_path = os.path.join(os.path.dirname(__file__), "movies.csv")

#  om filen finnes
if not os.path.exists(csv_path):
    sys.stdout.write(json.dumps([{"error": "movies.csv ikke funnet!"}]))
    sys.exit(1)  # Avslutt programmet

# prøv laste inn filen
try:
    movies = pd.read_csv(csv_path)
except Exception as e:
    sys.stdout.write(json.dumps([{"error": f"Kunne ikke laste movies.csv: {str(e)}"}]))
    sys.exit(1)

#  'genres' finnes i datasettet
if "genres" not in movies.columns:
    sys.stdout.write(json.dumps([{"error": "Mangler 'genres' i movies.csv"}]))
    sys.exit(1)

#  eventuelle kolonnenavnfeil (f.eks. "genres;" → "genres")
movies.rename(columns={"genres;": "genres"}, inplace=True)

#  kun relevante kolonner
movies = movies[["movieId", "title", "genres"]]

#  bruker med vurderinger
fictional_user_ratings = [
    "Interstellar (2014)", 
    "Inglorious Bastards (Quel maledetto treno blindato) (1978)", 
    "Hurt Locker, The (2008)", 
    "Harry Potter and the Half-Blood Prince (2009)", 
    "Inception (2010)", 
    "Fury (2014)", 
    "Matrix, The (1999)", 
    "Shrek (2001)", 
    "Fast and the Furious, The (2001)", 
    "xXx (2002)"
]

#  beste matchene for filmene i movies.csv
matched_movies = []
for title in fictional_user_ratings:
    best_match = process.extractOne(title, movies["title"], score_cutoff=80)  # 80% likhet
    if best_match:
        matched_movies.append(best_match[0])  # Legg til match

#  DataFrame med matchede filmer
user_ratings_df = movies[movies["title"].isin(matched_movies)].copy()

#  vurderinger til filmene
ratings_list = [5.0, 4.0, 5.0, 4.0, 4.5, 4.0, 5.0, 4.0, 2.0, 2.0]
ratings_list = ratings_list[:len(user_ratings_df)]  # listen er like lang som filmene
user_ratings_df["rating"] = ratings_list

# sjangre til numerisk format
movies["genres"] = movies["genres"].fillna("")  #  manglende sjangre
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(movies["genres"])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
movie_indices = pd.Series(movies.index, index=movies["movieId"])  # indeks

#  anbefalinger basert på fiktive vurderinger
def get_recommendations(movie_id, num_recommendations=5):
    if movie_id not in movie_indices:
        return pd.DataFrame()
    idx = movie_indices[movie_id]  # Finn indeks
    sim_scores = list(enumerate(cosine_sim[idx]))  # Hent likhetsscore
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:num_recommendations+1]  # Sorter
    return movies.iloc[[i[0] for i in sim_scores]][["movieId", "title"]]  # Returner anbefalinger

#  anbefalingene
recommended_movies = pd.DataFrame()
for movie_id in user_ratings_df["movieId"]:
    recommended_movies = pd.concat([recommended_movies, get_recommendations(movie_id)])

# fjerner duplikater og ta kun de 5 første anbefalingene
recommended_movies = recommended_movies.drop_duplicates().head(5)

# skriv ut JSON (Render trenger ren JSON-output)
sys.stdout.write(json.dumps(recommended_movies.to_dict(orient="records")))

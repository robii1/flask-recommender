# recommender_logic.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def get_recommendations():
    #  filmdata 
    df = pd.read_csv("movies.csv").fillna("")

    # sjangre til numerisk format
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(df["genres"])
    cosim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    #  film nr. 0 og henter 5 lignende filmer
    idx = 0
    sim_scores = list(enumerate(cosim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]

    # som JSON
    recommended_titles = [df.loc[i[0], "title"] for i in sim_scores]
    return [{"title": t} for t in recommended_titles]

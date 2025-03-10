import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def get_recommendations():
    #  movies.csv eksisterer?
    if not os.path.exists("movies.csv"):
        return [{"error": "movies.csv ikke funnet"}]

    try:
        # filmdata
        df = pd.read_csv("movies.csv").fillna("")
    except Exception as e:
        return [{"error": f"Kunne ikke laste movies.csv: {str(e)}"}]

    #  lignende filmer
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(df["genres"])
    cosim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    #  film nr. 0 og henter 5 lignende filmer
    idx = 0
    sim_scores = list(enumerate(cosim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]

    recommended_titles = [df.loc[i[0], "title"] for i in sim_scores]
    return [{"title": t} for t in recommended_titles]

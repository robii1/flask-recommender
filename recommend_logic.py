import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def get_recommendations():
    #  om movies.csv finnes
    csv_path = os.path.join(os.path.dirname(__file__), "movies.csv")
    
    if not os.path.exists(csv_path):
        return [{"error": "movies.csv ikke funnet!"}]

    # lese filen
    try:
        df = pd.read_csv(csv_path).fillna("")
    except Exception as e:
        return [{"error": f"Kunne ikke laste movies.csv: {str(e)}"}]

    #  "genres" finnes i datasettet
    if "genres" not in df.columns:
        return [{"error": "Mangler 'genres' i movies.csv"}]

    #  sjangre til numerisk format
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(df["genres"])
    cosim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    #  første film i datasettet og finn 5 lignende
    idx = 0
    sim_scores = list(enumerate(cosim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]

    #  anbefalte filmer
    recommended_titles = [df.loc[i[0], "title"] for i in sim_scores]
    return [{"title": t} for t in recommended_titles]

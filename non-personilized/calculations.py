import pandas as pd

links = pd.read_csv("C:/Users/Pratik/Desktop/Hybrid_movie/Hybrid-Movie-Recommendation/links.csv")
movies = pd.read_csv("C:/Users/Pratik/Desktop/Hybrid_movie/Hybrid-Movie-Recommendation/movies.csv")
ratings = pd.read_csv ("C:/Users/Pratik/Desktop/Hybrid_movie/Hybrid-Movie-Recommendation/ratings.csv")


df = pd.merge(movies, ratings, on = "movieId",how = "inner")
df = df.groupby(["movieId", "title"]).agg({"userId":"count", "rating":"mean"}).reset_index()
df = df[df["userId"]>100]
def findScore(x):
    v = x["userId"]
    m = 100
    r = x["rating"]
    c = 3.5
    score = ((v/(v+m))*r)+((m/(m+v))*c)
    return score

df["score"] = df.apply(findScore, axis  = 1)
df = df.sort_values(by = "score", ascending = False)
top_10_movies = df.head(10).to_json(orient = "records")
print(top_10_movies)
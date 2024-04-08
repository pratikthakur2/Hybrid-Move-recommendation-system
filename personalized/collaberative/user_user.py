import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


df = pd.read_csv("C:/Users/Pratik/Desktop/Hybrid_movie/Hybrid-Movie-Recommendation/ratings.csv")
# df is start with 1 let's start with 0
df["userId"] = df["userId"]-1

# movie Id is not in sequence first make in sequence
unique_movie_id = set(df["movieId"])
count = 0
movie_id_idx = {}
for i in unique_movie_id:
    movie_id_idx[i] = count
    count+=1


df["movie_idx"] = df["movieId"].apply(lambda x:movie_id_idx[x])

N = df["userId"].max()
M = df["movie_idx"].max()
print("Number of the user", N)
print("Number of the movie", M)

from collections import Counter
n = 3000
m = 500
movie_ids = [ids for ids,count in Counter(df["movie_idx"]).most_common(m)]
user_ids = [ids for ids,count in Counter(df["userId"]).most_common(n)]

# filter the user
df_selected = df[(df["userId"].isin(user_ids)) & (df["movie_idx"].isin(movie_ids))]

# reset the id
new_userid_map = {old:new for new, old in enumerate(user_ids)}
new_movieid_map = {old:new for new, old in enumerate(movie_ids)}

df_selected["userId"] = df_selected["userId"].apply(lambda x:new_userid_map[x])
df_selected["movie_idx"] = df_selected["movie_idx"].apply(lambda x:new_movieid_map[x])

N = df_selected["userId"].max()
M = df_selected["movie_idx"].max()
print("Number of the user", N+1)
print("Number of the movie", M+1)

df_selected.drop(["movieId", "timestamp"], axis = 1, inplace = True)

# creating user-item matrix
user_item_matrix = df_selected.pivot_table(index = "userId", columns = "movie_idx", values = "rating" )

user_item_matrix.fillna(0, inplace = True)

user_meanrating = {idx:rating for idx, rating in enumerate(user_item_matrix.mean(axis = 1))}

 

# mean center the data
mean_row =  np.mean(user_item_matrix.values, axis = 1).reshape(-1,1)
mean_centering = np.where(user_item_matrix>0,user_item_matrix-mean_row,
                          user_item_matrix)


cosine_matrix = pd.DataFrame(cosine_similarity(mean_centering))

# predict neighbour
def predict_neighbor(user):
    return cosine_matrix[user].sort_values(ascending = False)[1:21].to_dict()

# predict ratings
def predict_rating(user, item):
    neighb_sscore = predict_neighbor(user)
    neighb_itemrating  = {}
    for n in neighb_sscore.keys():
        neighb_itemrating[n] = mean_centering[int(n), int(item)]
    ratings = np.array(list(neighb_itemrating.values()))
    score = np.array(list(neighb_sscore.values()))
    predicted_rating = user_meanrating[user]+(np.dot(ratings, score)/sum(score))
    if predicted_rating>5:
        return 5
    else:
        return predicted_rating
    

 

def findTop10(user):
    dic = {}
    if user in user_ids:
        user = new_userid_map[user]
        for m in movie_ids:
            m = new_movieid_map[m]
            dic[m] = predict_rating(user, m)
        dic = sorted(dic.items(), key = lambda x:x[1], reverse = True)[:10]
    return dic

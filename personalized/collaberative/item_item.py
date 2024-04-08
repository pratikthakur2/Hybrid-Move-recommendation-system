import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter


df = pd.read_csv("C:/Users/Pratik/Desktop/Hybrid_movie/Hybrid-Movie-Recommendation/ratings.csv")

# user id is start from 1 do that from 0
df["userId"] = df["userId"] - 1

# movie_id not is in sequence 
# make movie_id in sequence
unique_movie_ids = set(df.movieId.values)
movie2idx = {}
count = 0
for movie_id in unique_movie_ids:
    movie2idx[movie_id] = count
    count += 1


# add them to the data frame
df['movie_idx'] = df.apply(lambda row: movie2idx[row.movieId], axis=1)


# drop the timestamp column no need of that
df.drop(["timestamp"], axis = 1, inplace = True)

N = df.userId.max()+1 #number of users
M = df.movie_idx.max()+1 # number of movies

# print(f"Number of users {N}")
# print(f"Number of movies {M}")

# lets found most common userID and movies there might be chance some user have not
# rate any movie and some movie did not get any rating
user_ids_count = Counter(df.userId)
movie__ids_count = Counter(df.movie_idx)

n = 10000 # select most common 10000 users who rated the atleeast 28 movie
m = 2000 # select most common that get atleast 180+ ratings
user_ids = [u for u,c in user_ids_count.most_common(n)]
movie_ids = [x for x,c in movie__ids_count.most_common(m)]


# filter the data
df_small = df[df["userId"].isin(user_ids) & df["movie_idx"].isin(movie_ids)].copy()

# need to make in sequece (UserId, movie_idx)
new_user_id_map = {old_id: new_id for new_id, old_id in enumerate(user_ids)}
new_movie_id_map = {old_id: new_id for new_id, old_id in enumerate(movie_ids)}
df_small["userId"] = df_small["userId"].apply(lambda x:new_user_id_map[x])
df_small["movie_idx"] = df_small["movie_idx"].apply(lambda x:new_movie_id_map[x])

# print("Max User Id: ", df_small.userId.max())
# print("Max Movie Id: ", df_small.movie_idx.max())


df = df_small.copy()
# a dictionery to tell us which users have rated which movies



user2movie = {}
# a dictionery to tell us which movies have been rated bu user
movie2user = {}
# a dictionery to look up ratings
usermovie2rating = {}
def update_user2movie_and_movie2user(row):
  # update user2movie dictionery
    if row.userId in user2movie:
        user2movie[row.userId].append(row.movie_idx)
    else:
        user2movie[row.userId] = [row.movie_idx]

  # update user2movie dictionery
    if row.movie_idx in movie2user:
        movie2user[row.movie_idx].append(row.userId)
    else:
        movie2user[row.movie_idx] = [row.userId]

    usermovie2rating[(row.userId, row.movie_idx)] = row.rating


df.apply(update_user2movie_and_movie2user, axis = 1)

item_user_matrix = df.pivot_table(index = "userId", columns = "movie_idx", values = "rating")

item_user_matrix = item_user_matrix.fillna(0)
item_user_matrix_corelation = item_user_matrix.corr()

def find_item_neghberhood(item_id, item_user_matrix_corelation):
    negbour = item_user_matrix_corelation[item_id].sort_values(ascending = False)[1:21]
    return negbour.to_dict()

def predict_rating(user_id, movie_id):
    ratings = []
    user_movie = set(user2movie[user_id])
    neighbours = find_item_neghberhood(movie_id,item_user_matrix_corelation) 
    neigbours_movie = set([m for m, r in neighbours.items()])
    common_movie = user_movie & neigbours_movie
    if len(common_movie)>0:
        for movie in common_movie:
            ratings.append(usermovie2rating[(user_id,movie)])
        corr = [r for m, r in neighbours.items() if m in common_movie]
        ratings = np.array(ratings)
#         print(ratings)
        corr = -1* np.array(corr)
#         print(corr)
        numerator = ratings.dot(corr)
#         print(numerator)
        denominater = corr.sum()
#         print(denominater)
        return round(numerator/denominater,3)
    else:
        return 0
    

def recomend_movie(user_id):
    movie_ratings = {}
    if user_id in user_ids:
        total_movie = set(list(movie2user.keys()))
        user_movie = set(user2movie[user_id])
    #     print(user_movie[:30])
        unseen_movie = list(total_movie.difference(user_movie))
        for movie in unseen_movie[:200]:
            movie_ratings[movie] = predict_rating(user_id, movie)
        return sorted(movie_ratings.items(), key = lambda x:x[1], reverse = True)[:10]
    return movie_ratings

# print(recomend_movie(379))




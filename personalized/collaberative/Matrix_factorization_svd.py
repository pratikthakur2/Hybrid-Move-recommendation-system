import pandas as pd
import numpy as np
from sklearn.decomposition import randomized_svd
import pickle

df = pd.read_csv("C:/Users/Pratik/Desktop/Hybrid_movie/Hybrid-Movie-Recommendation/ratings.csv")
df["userId"] = df["userId"]-1

movieidx = {}
count = 0
for i in set(df["movieId"]):
    movieidx[i] = count
    count+=1

df["movieidx"] = df["movieId"].apply(lambda x:movieidx[x])

x = df.groupby(["userId"])["rating"].count()
intrested_user = x[x>10].index


y = df.groupby(["movieidx"])["rating"].count()
popular_movie = y[y>50].index

df_small = df[df["userId"].isin(intrested_user) & df["movieidx"].isin(popular_movie)]

user_id_map = {old:new for new,old in enumerate(list(intrested_user))}
movie_id_map = {old:new for new, old in enumerate(list(popular_movie))}

df_small["userId"] = df_small["userId"].apply(lambda x:user_id_map[x])
df_small["movieidx"] = df_small["movieidx"].apply(lambda x:movie_id_map[x])

df_small.drop(["timestamp", "movieId"], axis = 1, inplace = True)

user_movie_matrix = df_small.pivot_table(index = "userId", columns = "movieidx",values =  "rating")


user_movie_matrix.fillna(0, inplace= True)

mean_data = user_movie_matrix.mean(axis = 1).values.reshape(-1,1)

mean_center_data = np.where(user_movie_matrix>0,user_movie_matrix-mean_data,user_movie_matrix)

user_movie_matrix.to_csv("mean_center_data.csv")

# print(user_movie_matrix)


U, sigma, vt = randomized_svd(mean_center_data,n_components = 100,n_iter=5,
                                  random_state=None)

final_dic = {
    "U":U,
    "Sigma":sigma,
    "vt":vt,
    "mean_data":mean_data,
    'intrested_user':intrested_user,
    'popular_movie':popular_movie,
    'user_id_map':user_id_map,
    'movie_id_map':movie_id_map
}

with open("svd_model.pkl", 'wb') as f:
    pickle.dump(final_dic, f)



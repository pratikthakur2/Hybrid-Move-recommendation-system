import pickle
import numpy as np
with open("svd_model.pkl", 'rb') as f:
    svd_data = pickle.load(f)


U = svd_data['U']
vt = svd_data["vt"]
sigma = svd_data["Sigma"]
mean_data = svd_data["mean_data"]
intrested_user = svd_data["intrested_user"]
user_id_map = svd_data["user_id_map"]
movie_id_map = svd_data["movie_id_map"]
popular_movie = svd_data["popular_movie"]



def predict(u,i):
    user = U[u,:]
    movie = vt[:,i]
    eigen = np.diag(sigma)
    return mean_data[u][0] + np.dot(user, np.dot(eigen, movie))



def find_recommendation(user):
    dic = {}
    if user in intrested_user:
        user = user_id_map[user]
        for m in movie_id_map:
            m = movie_id_map[m]
            rating = predict(user, m)
            if rating<=5:
                dic[m] = rating
            else:
                dic[m] = 5 
        dic = sorted(dic.items(), key = lambda x:x[1], reverse = True)[:10]
    return dic


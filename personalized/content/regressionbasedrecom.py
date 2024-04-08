#### Importing librareis
import numpy as np
import numpy.ma as ma
from numpy import genfromtxt
from collections import defaultdict
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import tabulate
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Dot
from keras.optimizers import Adam
import pickle
from keras.losses import MeanSquaredError





pd.set_option("display.precision", 1)



### load the dataset
link = pd.read_csv("/home/nooman/links.csv")
movies = pd.read_csv("/home/nooman/movies.csv")
ratings = pd.read_csv("/home/nooman/ratings.csv")
tags = pd.read_csv("/home/nooman/tags.csv")


df  = pd.merge(movies,ratings, on = "movieId", how = "inner")

df = df.drop(['timestamp'], axis = 1)

df["genres"] = df["genres"].str.split("|")

users = df.drop("movieId", axis = 1)

users = users.drop(["title"], axis = 1)

users_dic = defaultdict(dict)
for i in users.values:
    for j in i[0]:
        if j not in users_dic[i[1]]:
            users_dic[i[1]][j] = [i[2]]
        users_dic[i[1]][j].append(i[2])
        
dic = defaultdict(dict)
for i in users_dic.keys():
    for j in users_dic[i].keys():
        dic[i][j] = np.array(users_dic[i][j]).mean()
users_df = pd.DataFrame(dic).T.reset_index().rename(columns = {"index":"userId"}).fillna(0)


movies_dic = defaultdict(dict)
for i in df.values:
    for j in i[2]:
        movies_dic[i[0]][j] = 1


movies_df = pd.DataFrame(movies_dic).T.reset_index().rename(columns = {"index":"movieId"}).fillna(0)


movies_df = movies_df[['movieId', 'Adventure', 'Animation', 'Children', 'Comedy', 'Fantasy',
       'Romance', 'Action', 'Crime', 'Thriller', 'Mystery', 'Horror', 'Drama',
       'War', 'Western', 'Sci-Fi', 'Musical', 'Film-Noir', 'IMAX',
       'Documentary', '(no genres listed)']]


users = []
items = []
rating = []
for i in ratings.values:
    users.append(users_df[users_df["userId"] == i[0]].values[0][1:])
    items.append(movies_df[movies_df["movieId"] == i[1]].values[0][1:])
    rating.append(i[2])


users = np.array(users)
items = np.array(items)
rating = np.array(rating)



scaler = StandardScaler()
users_scaler = scaler.fit_transform(np.array(users))
item_scaler = scaler.fit_transform(np.array(items))


user_train, user_test = train_test_split(users_scaler, train_size = 0.8, shuffle  = True, random_state = 42)
item_train, item_test = train_test_split(item_scaler, train_size = 0.8, shuffle  = True, random_state = 42)
y_train, y_test = train_test_split(rating, train_size = 0.8, shuffle  = True, random_state = 42)


y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
from sklearn.preprocessing import MinMaxScaler
minmaxscaler = MinMaxScaler()
y_train_scaler = minmaxscaler.fit_transform(y_train)
y_test_scaler = minmaxscaler.transform(y_test)


print(user_train.shape)


user_NN = Sequential()
user_NN.add(Dense(256, activation="relu"))
user_NN.add(Dense(128, activation="relu"))
user_NN.add(Dense(20, activation="linear"))
user_NN.add(Dense(20, activation="linear"))

# Define item neural network
item_NN = Sequential()
item_NN.add(Dense(256, activation="relu"))
item_NN.add(Dense(128, activation="relu"))
item_NN.add(Dense(20, activation="linear"))

# Input layers
input_user = Input(shape=(user_train.shape[1],))
input_item = Input(shape=(item_train.shape[1],))

# Apply user and item neural networks
users_model = user_NN(input_user)
users_model = tf.keras.layers.Lambda(lambda x: tf.linalg.l2_normalize(x, axis=1))(users_model)

item_model = item_NN(input_item)
item_model = tf.keras.layers.Lambda(lambda x: tf.linalg.l2_normalize(x, axis=1))(item_model)

# Dot product layer
output = Dot(axes=1)([users_model, item_model])

# Create model
model = Model([input_user, input_item], output)
model.summary()




cost_fn = MeanSquaredError()
optimi = Adam(learning_rate=0.01)
model.compile(loss = cost_fn,
             optimizer = optimi,
             metrics = ["mean_squared_error"],
             run_eagerly = True)


tf.random.set_seed(1)
model.fit([user_train, item_train],y_train_scaler, epochs = 5)


print(model.evaluate([user_test, item_test],y_test_scaler))


movie_embeddings = {key:value for key, value in zip(movies_df["movieId"], movies_df.iloc[:,1:].values)}

model.save('content_based_filtering_model.keras')

with open('content_based_filtering_standardscaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

# Save the MinMaxScaler object
with open('content_based_filtering_minmaxcaler.pkl', 'wb') as minmaxscaler_file:
    pickle.dump(minmaxscaler, minmaxscaler_file)
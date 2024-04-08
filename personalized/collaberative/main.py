from flask import Flask, jsonify, request
app = Flask(__name__)
from user_user import findTop10
from item_item import recomend_movie
from svd_recommendation import find_recommendation

@app.route("/from_neigh_user", methods = ["GET", "POST"])
def recommend():
    if request.method == "POST":
        print(request.json)
        data = request.json
        user = data["user_id"]
        user_user = findTop10(user)
        item_item = recomend_movie(user)
        svd = find_recommendation(user)
        dic = {
            "user_user":user_user,
            "item_item":item_item,
            "svd":svd
        }
        return jsonify(dic)
    else:
        return jsonify({"msg":"pls upload your user-id"})
    
if __name__ == "__main__":
    app.run(debug = True)



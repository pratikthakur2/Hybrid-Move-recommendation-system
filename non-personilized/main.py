from flask import Flask, jsonify
app = Flask(__name__)
from calculations import top_10_movies

@app.route("/recommend", methods = ["GET"])
def recommend():
    return jsonify(top_10_movies)

if __name__ == "__main__":
    app.run(debug = True)
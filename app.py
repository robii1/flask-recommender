# app.py
from flask import Flask, jsonify
import recommend_logic

app = Flask(__name__)

# api-endepunkt for å hente anbefalinger
@app.route("/recommend_api")
def recommend_api():
    recommendations = recommend_logic.get_recommendations()
    return jsonify(recommendations)

if __name__ == "__main__":
    # flask-server lokalt på port 5000
    app.run(port=5000, debug=True)

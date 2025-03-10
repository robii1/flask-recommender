from flask import Flask, jsonify
import recommend_logic
import os  #os for å hente port fra Render

app = Flask(__name__)

# API-endepunkt for anbefalinger
@app.route("/recommend_api", methods=["GET"])
def recommend_api():
    recommendations = recommend_logic.get_recommendations()
    return jsonify(recommendations)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Render sin PORT eller 10000 som fallback
    app.run(host="0.0.0.0", port=port, debug=True)

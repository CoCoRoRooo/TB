from flask import Flask, request, jsonify
from flask_cors import CORS
from chatbot import chat_gpt, start_conversation, get_recommended_resources

app = Flask(__name__)
CORS(app)  # Active CORS pour toutes les routes


@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "")
    response = chat_gpt(user_input)

    return (
        jsonify({"response": response}),
        200,
    )  # Retourne la réponse JSON avec un code HTTP 200


@app.route("/start", methods=["POST"])
def open_conversation():
    """
    Lance la conversation avec une question initiale.
    """
    start_conversation()
    return (
        jsonify({"response": start_conversation()}),
        200,
    )


@app.route("/get_guides", methods=["POST"])
def get_guides():
    """
    Lance la conversation avec une question initiale.
    """
    return (
        jsonify({"response": get_recommended_resources()}),
        200,
    )


if __name__ == "__main__":
    app.run(debug=True)

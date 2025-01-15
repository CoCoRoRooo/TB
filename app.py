# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from chatbot import chat_gpt
from interface.user_interface import ask_user_question

app = Flask(__name__)
CORS(app)  # Active CORS pour toutes les routes


# Route pour générer une réponse en fonction du message utilisateur
@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get(
        "message", ""
    )  # Récupérer le message utilisateur depuis la requête
    context = "Tu es un assistant spécialisé dans les réparations électroniques. Aide l'utilisateur à identifier les ressources utiles pour effectuer les réparations répond sous forme de liste à points."
    response = chat_gpt(user_input, context)
    # Construire une réponse JSON
    response = {
        "response": response,
    }
    # Ajouter à la réponse le guide recommandé en fonction de la question utilisateur
    response.update(ask_user_question(user_input))

    return jsonify(response), 200  # Retourner la réponse JSON avec un code HTTP 200


if __name__ == "__main__":
    app.run(debug=True)

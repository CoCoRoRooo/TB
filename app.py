from flask import Flask, request, render_template, jsonify
from tokenizer import load_guides, index_guides, search_guides
from llm import generate_response

app = Flask(__name__)

# Charger et indexer les guides au démarrage du serveur
guides = load_guides("data/prepare_dataset/guides_dataset.json")
faiss_index, guide_texts, embed_model = index_guides(guides)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    user_query = data.get("query", "")

    if not user_query:
        return jsonify({"error": "Aucune question fournie"}), 400

    # Recherche des guides pertinents
    top_guides = search_guides(user_query, faiss_index, guide_texts, embed_model)

    # Génération de réponse avec OpenAI
    response = generate_response(user_query, top_guides)

    return jsonify({"query": user_query, "guides": top_guides, "response": response})


if __name__ == "__main__":
    app.run(debug=True)

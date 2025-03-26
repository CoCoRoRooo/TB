from flask import Flask, request, render_template, jsonify
from tokenizer import load_guides, index_guides
from llm import create_rag_chain

app = Flask(__name__)

# Charger et indexer les guides au démarrage du serveur
guides = load_guides("data/guides.json")
print(f"{len(guides)} guides chargés.")
retriever = index_guides(guides)

# Construire la chaîne RAG
rag_chain = create_rag_chain(retriever)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    user_query = data.get("query", "")

    if not user_query:
        return jsonify({"error": "Aucune question fournie"}), 400

    # Génération de réponse avec la chaîne RAG
    response = rag_chain.invoke(user_query)

    return jsonify({"query": user_query, "response": response})


if __name__ == "__main__":
    app.run(debug=True)

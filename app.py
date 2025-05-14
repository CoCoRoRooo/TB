from flask import Flask, render_template, request, jsonify
import logging
import time
from dotenv import load_dotenv
import os
import torch

# Import des modules personnalisés
from rag_chain import (
    initialize_rag_system,
    get_rag_chain,
    get_generate_queries,
    get_retrieval_chain,
)
from utils import format_documents

# Configuration du logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialiser l'application Flask
app = Flask(__name__)

# Charger les variables d'environnement
load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_KEY")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    start_time = time.time()
    data = request.json
    question = data.get("message", "")

    if not question:
        return jsonify({"error": "Message vide"}), 400

    logger.info(f"Question reçue: {question}")

    # Récupération des chaînes de traitement
    rag_chain = get_rag_chain()
    generate_queries = get_generate_queries()
    retrieval_chain = get_retrieval_chain()

    # Génération des requêtes alternatives
    queries = generate_queries.invoke(question)
    logger.info(f"Requêtes générées: {queries}")

    # Récupération des documents
    retrieved_docs = retrieval_chain.invoke(question)
    formatted_docs = format_documents(retrieved_docs)
    logger.info(f"Documents récupérés: {len(retrieved_docs)} documents")

    # Génération de la réponse
    response = rag_chain.invoke(question)

    elapsed_time = time.time() - start_time
    logger.info(f"Réponse générée en {elapsed_time:.2f} secondes")

    return jsonify(
        {
            "response": response,
            "queries": queries,
            "documents": formatted_docs,
            "processing_time": f"{elapsed_time:.2f} secondes",
        }
    )


if __name__ == "__main__":
    # Ne pas initialiser deux fois avec le reloader
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        with app.app_context():
            initialize_rag_system()

    # Vérifier si CUDA est disponible
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Utilisation du périphérique: {device}")

    app.run(debug=True, host="0.0.0.0", port=5000)

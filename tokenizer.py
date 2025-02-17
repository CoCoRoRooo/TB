import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer


# Charger les guides depuis un fichier JSON
def load_guides(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        guides = json.load(f)
    return guides


# Convertir les guides en vecteurs
def index_guides(
    guides, model_name="./sbert_fine_tuned_symptoms"
):  # Utilisation du modèle Fine-tune
    model = SentenceTransformer(model_name)
    # texts = [f"{g['dataType']} - {g['symptoms']} {(g['url'])}" for g in guides]
    texts = [
        f"{g['dataType']} - {g['title']}; {g['general_solution']} {(g['url'])}"
        for g in guides
    ]

    # Créer des embeddings
    embeddings = model.encode(texts, convert_to_numpy=True)

    # Stocker les embeddings dans FAISS
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)  # L2 = Euclidean distance
    index.add(embeddings)

    return index, texts, model


def search_guides(query, faiss_index, guide_texts, model, top_k=3):
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = faiss_index.search(query_embedding, top_k)

    results = [guide_texts[i] for i in indices[0]]
    return results

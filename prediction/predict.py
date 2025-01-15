import torch
import numpy as np
from models.bert_vectorizer import BertVectorizer
from models.similarity import calculate_similarity


def predict_guides(
    user_question, guides, embedding_file="guide_embeddings.pt", top_n=3
):
    """
    Prédit les meilleurs guides en fonction de la similarité avec la question utilisateur.

    Args:
        user_question (str): La question posée par l'utilisateur.
        guides (list): Liste des guides.
        embedding_file (str): Chemin vers le fichier des embeddings précalculés.
        top_n (int): Nombre de guides à retourner.

    Returns:
        list: Les `top_n` guides les plus similaires.
    """
    # Charger les embeddings des guides
    guide_vectors = torch.load(embedding_file)

    # Vectoriser la question utilisateur
    bert_vectorizer = BertVectorizer()
    user_question_embedding = bert_vectorizer.vectorize_texts([user_question])

    # Calculer les similarités
    similarities = calculate_similarity(user_question_embedding, guide_vectors)

    # Obtenir les indices des `top_n` meilleures similarités
    top_indices = np.argsort(similarities.flatten())[-top_n:][::-1]

    # Retourner les `top_n` guides
    return [guides[i] for i in top_indices]

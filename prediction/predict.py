import torch
import numpy as np
from models.sbert_vectorizer import SBertVectorizer
from models.similarity import calculate_similarity


def predict_guides(
    user_question,
    guides,
    embedding_file="guide_embeddings.pt",
    top_n=3,
    similarity_threshold=0.7,
):
    """
    Prédit les meilleurs guides en fonction de la similarité avec la question utilisateur,
    uniquement si leur similarité dépasse un seuil minimum.

    Args:
        user_question (str): La question posée par l'utilisateur.
        guides (list): Liste des guides.
        embedding_file (str): Chemin vers le fichier des embeddings précalculés.
        top_n (int): Nombre de guides à retourner.
        similarity_threshold (float): Seuil minimum de similarité pour retourner un guide.

    Returns:
        list: Les guides les plus similaires si leur similarité dépasse le seuil, sinon liste vide.
    """
    # Charger les embeddings des guides
    guide_vectors = torch.load(embedding_file)

    # Vectoriser la question utilisateur
    sbert_vectorizer = SBertVectorizer()
    user_question_embedding = sbert_vectorizer.vectorize_texts([user_question])

    # Calculer les similarités
    similarities = calculate_similarity(user_question_embedding, guide_vectors)

    # Obtenir les indices des `top_n` meilleures similarités
    top_indices = np.argsort(similarities.flatten())[-top_n:][::-1]

    # Filtrer les guides en fonction du seuil de similarité
    relevant_guides = []
    for i in top_indices:
        if similarities.flatten()[i] >= similarity_threshold:
            relevant_guides.append(guides[i])

    # Retourner les guides pertinents (s'ils existent)
    return relevant_guides

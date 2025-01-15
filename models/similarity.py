from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F


def calculate_similarity(question_vector, guide_vectors):
    # On normalise le vecteur de la question tout comme les vecteurs des guides l'ont été lors de la sauvegarde "save_embeddings.py"
    question_vector = F.normalize(question_vector, p=2, dim=1)

    return cosine_similarity(question_vector.cpu().numpy(), guide_vectors.cpu().numpy())

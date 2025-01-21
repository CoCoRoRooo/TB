from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F


def calculate_similarity(chatbot_infos_vector, guide_vectors):
    # On normalise le vecteur de réponse du chatbot tout comme les vecteurs des guides l'ont été lors de la sauvegarde "save_embeddings.py"
    chatbot_infos_vector = F.normalize(chatbot_infos_vector, p=2, dim=1)

    return cosine_similarity(
        chatbot_infos_vector.cpu().numpy(), guide_vectors.cpu().numpy()
    )

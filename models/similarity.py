from sklearn.metrics.pairwise import cosine_similarity


def calculate_similarity(question_vector, guide_vectors):
    return cosine_similarity(question_vector.cpu().numpy(), guide_vectors.cpu().numpy())

import torch
import json
from sbert_vectorizer import SBertVectorizer
from preprocessing import preprocess_texts
import torch.nn.functional as F


def save_guide_embeddings(guides, output_file="guide_embeddings.pt"):
    # Extraire les métadonnées des guides
    guides_metadata = [
        f"{guide['category']} {guide['subject']} {guide['title']}" for guide in guides
    ]
    processed_texts = preprocess_texts(guides_metadata)

    # Initialisation du vectoriseur SBERT
    sbert_vectorizer = SBertVectorizer()
    embeddings = sbert_vectorizer.vectorize_texts_in_batches(
        processed_texts, batch_size=16
    )

    # Normalisation des embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)

    # Sauvegarder les embeddings
    torch.save(embeddings, output_file)
    print(f"Embeddings saved to {output_file}")


if __name__ == "__main__":
    with open("guides.json", "r") as f:
        guides = json.load(f)
    save_guide_embeddings(guides)

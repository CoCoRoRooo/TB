import torch
import json
from bert_vectorizer import BertVectorizer
from preprocessing import preprocess_texts


def save_guide_embeddings(guides, output_file="guide_embeddings.pt"):
    guide_title = [guide["title"] for guide in guides]
    processed_texts = preprocess_texts(guide_title)

    bert_vectorizer = BertVectorizer()
    embeddings = bert_vectorizer.vectorize_texts_in_batches(
        processed_texts, batch_size=16
    )

    torch.save(embeddings, output_file)
    print(f"Embeddings saved to {output_file}")


if __name__ == "__main__":
    with open("guides.json", "r") as f:
        guides = json.load(f)
    save_guide_embeddings(guides)

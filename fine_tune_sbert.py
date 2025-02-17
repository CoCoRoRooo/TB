import json
import torch
import random
import numpy as np
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
from sentence_transformers.evaluation import SentenceEvaluator
from torch.utils.data import DataLoader

# Vérifier si CUDA est dispo
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Charger le dataset
with open("data/prepare_dataset/guides_dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Préparer les paires (symptômes, titres)
samples = []
for item in data:
    symptoms = item.get("symptoms", "")
    title = item.get("title", "")

    if symptoms and title:  # Vérifie que les valeurs existent
        symptoms = symptoms.replace("Symptoms: ", "").strip()
        title = title.replace("Title: ", "").strip()
        samples.append((symptoms, title))


print(
    samples[:50]
)  # Voir les 50 premières paires symptômes -> titres de guide (cibles)

# Mélanger les données pour éviter l'ordre du dataset
random.shuffle(samples)

# Séparer en 80% train / 20% eval
split_idx = int(len(samples) * 0.8)
train_data = samples[:split_idx]
eval_data = samples[split_idx:]

# Préparer les données pour SBERT
train_samples = [InputExample(texts=[s, p]) for s, p in train_data]
eval_samples = [InputExample(texts=[s, p]) for s, p in eval_data]

# Charger SBERT
model_name = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(model_name, device=device)

# Définir la loss function
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=16)
train_loss = losses.MultipleNegativesRankingLoss(model)


# Évaluation de similarité
class SimilarityEvaluator(SentenceEvaluator):
    def __init__(self, eval_pairs, name="SimilarityEvaluator"):
        super().__init__()
        self.eval_pairs = eval_pairs
        self.name = name

    def compute_similarity(self, model):
        scores = []
        for s, p in self.eval_pairs:
            s_embed = model.encode(s, convert_to_tensor=True)
            p_embed = model.encode(p, convert_to_tensor=True)
            sim = torch.nn.functional.cosine_similarity(s_embed, p_embed, dim=0).item()
            scores.append(sim)
        return scores

    def __call__(self, model, output_path=None, epoch=-1, steps=-1):
        scores = self.compute_similarity(model)
        mean_score = np.mean(scores)
        print(f"Epoch {epoch}, Similarité moyenne = {mean_score:.4f}")
        return mean_score


# Initialiser l'évaluateur
evaluator = SimilarityEvaluator(eval_data)

# Entraînement avec évaluation
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    evaluator=evaluator,
    epochs=5,
    warmup_steps=100,
    show_progress_bar=True,
)

# Sauvegarde du modèle entraîné
model.save("sbert_fine_tuned_symptoms")

# Évaluer la similarité finale
final_scores = evaluator.compute_similarity(model)
final_mean_score = np.mean(final_scores)
print(f"\nSimilarité moyenne finale après entraînement: {final_mean_score:.4f}")

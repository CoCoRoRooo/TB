from sentence_transformers import SentenceTransformer
import torch
from torch.utils.data import DataLoader


class SBertVectorizer:
    def __init__(self, model_name="all-MiniLM-L6-v2", device=None):
        """
        Initialise le modèle SBERT.

        Args:
            model_name (str): Le nom du modèle SBERT à utiliser.
            Par défaut, on utilise "all-MiniLM-L6-v2" (léger et performant).
            device (str): Le dispositif sur lequel exécuter le modèle, "cuda" ou "cpu".
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_name, device=self.device)

    def vectorize_texts(self, texts):
        """
        Vectorise les textes en utilisant SBERT.

        Args:
            texts (list): Une liste de textes à vectoriser.

        Returns:
            torch.Tensor: Les embeddings générés par SBERT.
        """
        embeddings = self.model.encode(texts, convert_to_tensor=True)
        return embeddings

    def vectorize_texts_in_batches(self, texts, batch_size=16):
        """
        Vectorise les textes en lots à l'aide de SBERT.

        Args:
            texts (list): Une liste de textes à vectoriser.
            batch_size (int): La taille du lot pour la vectorisation.

        Returns:
            torch.Tensor: Les embeddings générés par SBERT.
        """
        embeddings_list = []
        dataloader = DataLoader(texts, batch_size=batch_size, shuffle=False)
        for batch in dataloader:
            batch_embeddings = self.model.encode(batch, convert_to_tensor=True)
            embeddings_list.append(batch_embeddings)

        return torch.cat(embeddings_list, dim=0)

import torch
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader


class BertVectorizer:
    def __init__(self, model_name="distilbert-base-uncased", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)

    def vectorize_texts(self, texts):
        tokens = self.tokenizer(
            texts, padding=True, truncation=True, max_length=128, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**tokens)
            embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings.cpu()

    def vectorize_texts_in_batches(self, texts, batch_size=16):
        dataloader = DataLoader(texts, batch_size=batch_size, shuffle=False)
        embeddings_list = []
        for batch in dataloader:
            tokens = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                outputs = self.model(**tokens)
                embeddings = outputs.last_hidden_state.mean(dim=1)
                embeddings_list.append(embeddings.cpu())
        return torch.cat(embeddings_list, dim=0)

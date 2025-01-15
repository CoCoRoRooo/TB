import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def preprocess_texts(texts):
    """
    Prétraite une liste de textes : mise en minuscules, suppression des ponctuations.

    Args:
        texts (list): Liste de textes à prétraiter.

    Returns:
        list: Liste des textes prétraités.
    """
    stop_words = set(stopwords.words("english"))

    def clean_text(text):
        text = text.lower()
        text = text.translate(
            str.maketrans("", "", string.punctuation)
        )  # Supprimer la ponctuation
        words = word_tokenize(text)
        words = [word for word in words if word not in stop_words]
        return " ".join(words)

    return [clean_text(text) for text in texts]

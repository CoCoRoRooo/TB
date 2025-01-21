# Chatbot de Recommandation pour la Réparation Électronique

Ce projet implémente un chatbot capable de fournir des recommandations de guides pour la réparation électronique en utilisant des modèles de machine learning, des embeddings SBERT et des algorithmes de similarité.

## Fonctionnalités Principales

- Extraction des guides de réparation depuis l'API iFixit.
- Vectorisation des titres de guides à l'aide de **all-MiniLM-L6-v2**.
- Prétraitement des textes (nettoyage, suppression des stopwords).
- Calcul de similarité entre la dernière réponse du chatbot et les guides disponibles.
- Traduction automatique des réponse du chatbot pour une compatibilité multi-langue (via **deep-translator**).
- Interaction conversationnelle via un chatbot OpenAI, avec un contexte défini et un historique des messages.
- Recommandation des guides les plus pertinents uniquement s'ils atteignent un seuil minimum de similarité.
- La recherche de guides repose sur la dernière réponse générée par l'agent GPT, et s'affine avec le temps grâce aux interactions avec l'utilisateur, maximisant ainsi les chances de proposer le guide le plus adapté.
- Interface utilisateur interactive avec un frontend HTML pour communiquer avec l'API Flask.

## Modifications Récentes

### 1. Recherche Vidéo YouTube pour les Guides Retournés (ytb_videos.py)

Ajout d'un module permettant de rechercher des vidéos YouTube pertinentes pour chaque guide recommandé.

#### Fonctionnalités du Module `ytb_videos.py`

- Utilisation de l'API YouTube Data v3 pour effectuer des recherches basées sur les titres et catégories des guides.
- Retourne les 3 vidéos les plus pertinentes par guide, incluant leur titre, description, URL et miniature.

#### Exemple de Code :

```python
from googleapiclient.discovery import build
from dotenv import load_dotenv
import os

# Configuration de l'API YouTube
load_dotenv()

YTB_DATA_API_V3_KEY = os.getenv("YTB_DATA_API_V3_KEY")
youtube = build("youtube", "v3", developerKey=YTB_DATA_API_V3_KEY)

def search_youtube_videos(keyword, max_results=3):
    """
    Recherche des vidéos YouTube basées sur un mot-clé.

    :param keyword: Mot-clé pour la recherche.
    :param max_results: Nombre maximum de résultats à récupérer.
    :return: Liste des vidéos trouvées.
    """
    try:
        print(f"Recherche de vidéos YouTube pour : {keyword}")
        search_response = (
            youtube.search()
            .list(
                q=keyword,
                part="snippet",
                type="video",
                maxResults=max_results,
            )
            .execute()
        )

        videos = []
        for item in search_response.get("items", []):
            video = {
                "video_id": item["id"]["videoId"],
                "title": item["snippet"]["title"],
                "description": item["snippet"]["description"],
                "channel_title": item["snippet"]["channelTitle"],
                "published_at": item["snippet"]["publishedAt"],
                "thumbnail_url": item["snippet"]["thumbnails"]["default"]["url"],
                "url": f"https://www.youtube.com/watch?v={item['id']['videoId']}"
            }
            videos.append(video)

        return videos

    except Exception as e:
        print(f"Erreur lors de la recherche : {e}")
        return []
```

### 2. Filtrage des guides basé sur un seuil de similarité (predict.py)

La fonction **predict_guides** a été mise à jour pour retourner uniquement les guides qui dépassent un seuil minimum de similarité (paramètre `similarity_threshold`). Si aucun guide ne dépasse ce seuil, une liste vide est retournée. Cela garantit que seules les recommandations pertinentes sont affichées.

#### Code Exemple :

```python
def predict_guides(
    chatbot_infos,
    guides,
    embedding_file="guide_embeddings.pt",
    top_n=3,
    similarity_threshold=0.7,
):
    """
    Prédit les meilleurs guides en fonction de la similarité avec la dernière réponse du chatbot,
    uniquement si leur similarité dépasse un seuil minimum.

    Args:
        chatbot_infos (str): La dérnière réponse donné par le chatbot.
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
    chatbot_infos_embedding = sbert_vectorizer.vectorize_texts([chatbot_infos])

    # Calculer les similarités
    similarities = calculate_similarity(chatbot_infos_embedding, guide_vectors)

    # Obtenir les indices des `top_n` meilleures similarités
    top_indices = np.argsort(similarities.flatten())[-top_n:][::-1]

    # Filtrer les guides en fonction du seuil de similarité
    relevant_guides = []
    for i in top_indices:
        if similarities.flatten()[i] >= similarity_threshold:
            relevant_guides.append(guides[i])

    # Retourner les guides pertinents (s'ils existent)
    return relevant_guides
```

### 3. Historique de conversation et contexte (chatbot.py)

Le chatbot conserve un historique des messages grâce à **ConversationManager**. Le dernier message généré par l'assistant est utilisé comme base pour prédire les guides pertinents. Cela permet d'affiner la recherche avec le temps et les interactions utilisateur.

#### Principales Fonctionnalités :

- Contexte initial défini pour le chatbot, spécifiant son rôle et ses objectifs.
- La fonction **get_recommended_resources** utilise le dernier message de l'assistant pour générer des recommandations basées sur l'interaction précédente.

#### Code Exemple :

```python
from models.conversation_manager import ConversationManager

def get_recommended_resources():
    resources_text_data = ""
    for message in reversed(conversation_manager.get_history()):
        if message["role"] == "assistant":
            resources_text_data = message["content"]
            break

    return get_resources(resources_text_data)
```

### 4. Gestion de l'historique (conversation_manager.py)

Un nouveau module, **ConversationManager**, gère l'historique de la conversation. Cela permet au chatbot de maintenir un contexte cohérent tout au long de l'interaction.

#### Code Exemple :

```python
class ConversationManager:
    def __init__(self):
        self.history = []

    def add_message(self, role, content):
        self.history.append({"role": role, "content": content})

    def get_history(self):
        return self.history

    def clear_history(self):
        self.history = []
```

## Structure du Projet

```
CHATBOT/
├── app.py                        # Application Flask pour interagir avec l'utilisateur
├── chatbot.py                    # Interaction avec l'API OpenAI
├── data/
│   ├── guides.json               # Fichier JSON contenant les guides extraits
│   ├── guides.py                 # Extraction des guides depuis l'API iFixit
├── models/
│   ├── sbert_vectorizer.py       # Vectorisation des textes avec SBERT
│   ├── load_data.py              # Chargement des données des guides
│   ├── preprocessing.py          # Prétraitement des textes
│   ├── save_embeddings.py        # Génération et sauvegarde des embeddings
│   ├── similarity.py             # Calcul des similarités entre vecteurs
├── prediction/
│   ├── predict.py                # Prédiction des guides pertinents
│   ├── ytb_videos.py             # Retourne les 3 vidéos les plus pertinentes pour chaque guide retourné
│   ├── recommendation_system.py  # Cet orchestrateur charge les guides et appels predict_guides
├── interface/
│   ├── user_interface.py         # Interface utilisateur pour traduire les questions et obtenir des recommandations
│   ├── chatbot.html              # Interface utilisateur (frontend)
├── utils/
│   ├── conversation_manager.py   # Gestion de l'historique des conversations
```

### Description des Fichiers

- **`sbert_vectorizer.py`** : Ce module contient une classe pour vectoriser des textes en utilisant SBert. Les textes sont transformés en vecteurs numériques via le modèle `all-MiniLM-L6-v2`. Cette vectorisation est au cœur de l'algorithme de recommandation, car elle permet de représenter les textes de manière sémantique. Il offre également une méthode pour traiter les textes par lots, améliorant ainsi l'efficacité lors du traitement de grandes quantités de données.

  ```python
  class SBertVectorizer:
      def __init__(self, model_name="all-MiniLM-L6-v2", device=None):
          # Initialisation du modèle et du tokenizer
  ```

- **`preprocessing.py`** : Implémente le nettoyage des textes, notamment la mise en minuscules, la suppression des ponctuations et des mots inutiles (stopwords). Ces étapes sont cruciales pour garantir que les embeddings générés soient représentatifs et cohérents.

  ```python
  def preprocess_texts(texts):
      # Fonction de prétraitement
  ```

- **`save_embeddings.py`** : Génère les embeddings des titres des guides en utilisant la classe `SBertVectorizer`. Ces vecteurs sont ensuite sauvegardés pour être utilisés dans les calculs de similarité. Ce fichier est exécuté après avoir extrait et nettoyé les données des guides.

  ```python
  def save_guide_embeddings(guides):
      # Calcul et sauvegarde des embeddings
  ```

- **`similarity.py`** : Fournit une fonction pour calculer la similarité cosinus entre les vecteurs, une mesure essentielle pour évaluer la pertinence des guides par rapport à une question utilisateur.

  ```python
  def calculate_similarity(chatbot_infos_vector, guide_vectors):
      # Calcul de similarité cosinus
  ```

- **`predict.py`** : Combine plusieurs étapes, dont la vectorisation de la question utilisateur et le calcul de similarité, pour identifier les guides les plus pertinents. Il exploite les embeddings précalculés pour des prédictions rapides.

  ```python
  def predict_guides(user_question):
      # Pipeline de prédiction
  ```

- **`chatbot.py`** : Permet une interaction avec l'API OpenAI pour enrichir les recommandations avec des réponses textuelles générées dynamiquement. Cela offre une dimension conversationnelle au chatbot.

  ```python
  def chat_gpt(prompt):
      # Fonction d'interaction avec OpenAI
  ```

- **`user_interface.py`** : Traduit les réponses du chatbot en anglais, exécute les prédictions et affiche les résultats de manière conviviale. Ce module est un pont entre l'utilisateur final et le backend du système.

  ```python
  def get_resources(chatbot_infos):
      # Interaction avec l'utilisateur
  ```

- **`ytb_videos.py`** : Recherche des vidéos YouTube associées aux guides retournés.

- **`recommendation_system.py`** : Orchestration du chargement des guides et des appels à `predict_guides`.

- **`conversation_manager.py`** : Gestion de l'historique des conversations avec le chatbot.

## Installation

1. Clonez ce dépôt :

   ```bash
   git clone <url-du-dépôt>
   ```

### 2. Installez les dépendances :

```bash
pip install flask flask-cors torch transformers sentence-transformers nltk scikit-learn deep-translator python-dotenv google-api-python-client
```

#### Explication des dépendances :

- **`flask`** : Framework léger pour créer des applications web en Python. Il est utilisé ici pour exposer un serveur API pour interagir avec le chatbot.
- **`flask-cors`** : Permet de gérer les en-têtes CORS (Cross-Origin Resource Sharing), ce qui est utile pour autoriser l'application web à interagir avec d'autres domaines, par exemple en utilisant un frontend sur un domaine différent.

- **`torch`** : La bibliothèque PyTorch, utilisée pour travailler avec des réseaux neuronaux et des modèles de deep learning. C'est une dépendance essentielle pour utiliser des modèles comme SBERT ou d'autres architectures basées sur des transformers.

- **`transformers`** : Une bibliothèque de Hugging Face qui fournit des modèles préentraînés et des outils pour travailler avec des architectures de transformers, comme BERT, GPT, et bien d'autres.

- **`sentence-transformers`** : Cette bibliothèque étend `transformers` pour permettre des embeddings de phrases avec des modèles comme SBERT. Elle est spécialement utile pour la recherche de similarités entre des phrases ou des documents.

- **`nltk`** : La bibliothèque Natural Language Toolkit (NLTK) offre des outils pour le traitement du langage naturel (tokenisation, nettoyage de texte, stopwords, etc.). Elle est utilisée ici pour prétraiter les textes avant de les vectoriser avec SBERT.

- **`scikit-learn`** : Une bibliothèque pour le machine learning. Elle est utilisée ici principalement pour la fonction de calcul de similarité (par exemple, la similarité cosinus entre les embeddings).

- **`deep-translator`** : Une bibliothèque pour effectuer des traductions automatiques, pour traduire les questions des utilisateurs, car les guides récupérés depuis l'API d'`iFixit` sont en anglais.

- **`python-dotenv`** : Permet de charger des variables d'environnement depuis un fichier `.env`. Cela permet de garder des informations sensibles (comme des clés API) hors du code source, améliorant ainsi la sécurité et la portabilité de l'application.

- **`google-api-python-client`** : Fournit une interface Python pour interagir avec les API Google, comme YouTube Data API v3. Dans ce projet, il est utilisé pour rechercher des vidéos YouTube pertinentes en fonction des guides recommandés.

3. Téléchargez les ressources nécessaires pour NLTK :

   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')
   ```

## Utilisation

1. **Récupération des Guides :**
   Exécutez le script pour récupérer les guides depuis l'API iFixit et les enregistrer dans `guides.json` :

   ```bash
   python guides.py
   ```

2. **Génération des Embeddings :**
   Calculez et sauvegardez les embeddings des guides pour accélérer les prédictions :

   ```bash
   python models/save_embeddings.py
   ```

3. **Lancement de l'Application Flask :**
   Démarrez l'application pour interagir avec le chatbot :

   ```bash
   python app.py
   ```

4. **Interface Web :**
   Ouvrez le fichier `chatbot.html` dans un navigateur pour interagir avec le chatbot. Ce fichier envoie des requêtes à l'API Flask et affiche les réponses sous forme interactive.

## Fonctionnement en Détail

### Embeddings SBERT et Génération des Recommandations

1. **Génération des Embeddings (save_embeddings.py)**
   Les titres des guides sont vectorisés en embeddings à l'aide de SBERT. Ces vecteurs sont normalisés et stockés dans un fichier pour être utilisés dans les calculs de similarité.

2. **Prétraitement des Textes (preprocessing.py)**
   Avant la vectorisation, les textes sont nettoyés pour améliorer leur qualité.

3. **Prédiction (predict.py)**
   Une fois la dernière réponse du chatbot reçu, elle est traduite en anglais, vectorisée, puis comparée aux embeddings des guides pour calculer les similarités. Seuls les guides dépassant un seuil de similarité sont retournés.

### Pipeline Global

1. **Traduction (user_interface.py)** : La dernière réponse du chatbot est traduite pour assurer une compatibilité avec les modèles en anglais.
2. **Vectorisation (sbert_vectorizer.py)** : La question traduite est convertie en vecteur.
3. **Calcul de Similarité (similarity.py)** : La similarité entre le vecteur question et les embeddings des guides est calculée.
4. **Recommandations (recommendation_system.py)** : Les guides les plus pertinents sont renvoyés.

## Points Clés à Vérifier

- **API OpenAI :** Assurez-vous de configurer une clé API valide dans `chatbot.py`.
- **CUDA :** Si vous disposez d'un GPU, PyTorch l'utilisera automatiquement pour accélérer les calculs.
- **Données iFixit :** Vérifiez régulièrement que les données de l'API iFixit sont mises à jour dans `guides.json`.

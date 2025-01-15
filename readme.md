# Chatbot de Recommandation pour la Réparation Électronique

Ce projet implémente un chatbot capable de fournir des recommandations de guides pour la réparation électronique en utilisant des modèles de machine learning, des embeddings BERT et des algorithmes de similarité.

## Fonctionnalités Principales

- Extraction des guides de réparation depuis l'API iFixit.
- Vectorisation des titres de guides à l'aide de `distilbert-base-uncased`.
- Prétraitement des textes (nettoyage, suppression des stopwords).
- Calcul de similarité entre une question utilisateur et les guides disponibles.
- Traduction automatique des questions utilisateur pour une compatibilité multi-langue (via `deep-translator`).
- Interface utilisateur interactive avec un frontend HTML pour communiquer avec l'API Flask.

## Structure du Projet

```
CHATBOT/
├── app.py                        # Application Flask pour interagir avec l'utilisateur
├── chatbot.py                    # Interaction avec l'API OpenAI
├── guides.py                     # Extraction des guides depuis l'API iFixit
├── guides.json                   # Fichier JSON contenant les guides extraits
├── chatbot.html                  # Interface utilisateur (frontend)
├── models/
│   ├── bert_vectorizer.py        # Vectorisation des textes avec BERT
│   ├── load_data.py              # Chargement des données des guides
│   ├── preprocessing.py          # Prétraitement des textes
│   ├── save_embeddings.py        # Génération et sauvegarde des embeddings
│   ├── similarity.py             # Calcul des similarités entre vecteurs
├── prediction/
│   ├── predict.py                # Prédiction des guides pertinents
│   ├── recommendation_system.py  # Système de recommandation basé sur la similarité
├── interface/
│   ├── user_interface.py         # Interface utilisateur pour traduire les questions et obtenir des recommandations
```

### Description des Fichiers

- **`bert_vectorizer.py`** : Ce module contient une classe pour vectoriser des textes en utilisant BERT. Les textes sont transformés en vecteurs numériques via le modèle `distilbert-base-uncased`. Cette vectorisation est au cœur de l'algorithme de recommandation, car elle permet de représenter les textes de manière sémantique. Il offre également une méthode pour traiter les textes par lots, améliorant ainsi l'efficacité lors du traitement de grandes quantités de données.

  ```python
  class BertVectorizer:
      def __init__(self, model_name="distilbert-base-uncased"):
          # Initialisation du modèle et du tokenizer
  ```

- **`preprocessing.py`** : Implémente le nettoyage des textes, notamment la mise en minuscules, la suppression des ponctuations et des mots inutiles (stopwords). Ces étapes sont cruciales pour garantir que les embeddings générés soient représentatifs et cohérents.

  ```python
  def preprocess_texts(texts):
      # Fonction de prétraitement
  ```

- **`save_embeddings.py`** : Génère les embeddings des titres des guides en utilisant la classe `BertVectorizer`. Ces vecteurs sont ensuite sauvegardés pour être utilisés dans les calculs de similarité. Ce fichier est exécuté après avoir extrait et nettoyé les données des guides.

  ```python
  def save_guide_embeddings(guides):
      # Calcul et sauvegarde des embeddings
  ```

- **`similarity.py`** : Fournit une fonction pour calculer la similarité cosinus entre les vecteurs, une mesure essentielle pour évaluer la pertinence des guides par rapport à une question utilisateur.

  ```python
  def calculate_similarity(question_vector, guide_vectors):
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

- **`user_interface.py`** : Traduit les questions utilisateur en anglais, exécute les prédictions et affiche les résultats de manière conviviale. Ce module est un pont entre l'utilisateur final et le backend du système.

  ```python
  def ask_user_question(user_question):
      # Interaction avec l'utilisateur
  ```

## Installation

1. Clonez ce dépôt :

   ```bash
   git clone <url-du-dépôt>
   cd CHATBOT
   ```

2. Installez les dépendances :

   ```bash
   pip install flask flask-cors torch transformers nltk scikit-learn deep-translator
   ```

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

### Embeddings BERT et Génération des Recommandations

1. **Génération des Embeddings (save_embeddings.py)**
   Les titres des guides sont vectorisés en embeddings à l'aide de BERT. Ces vecteurs sont normalisés et stockés dans un fichier pour être utilisés dans les calculs de similarité.

2. **Prétraitement des Textes (preprocessing.py)**
   Avant la vectorisation, les textes sont nettoyés pour améliorer leur qualité.

3. **Prédiction (predict.py)**
   Lorsqu'une question est posée, elle est traduite en anglais, vectorisée, puis comparée aux embeddings des guides pour calculer les similarités.

### Pipeline Global

1. **Traduction (user_interface.py)** : La question est traduite pour assurer une compatibilité avec les modèles en anglais.
2. **Vectorisation (bert_vectorizer.py)** : La question traduite est convertie en vecteur.
3. **Calcul de Similarité (similarity.py)** : La similarité entre le vecteur question et les embeddings des guides est calculée.
4. **Recommandations (recommendation_system.py)** : Les guides les plus pertinents sont renvoyés.

## Points Clés à Vérifier

- **API OpenAI :** Assurez-vous de configurer une clé API valide dans `chatbot.py`.
- **CUDA :** Si vous disposez d'un GPU, PyTorch l'utilisera automatiquement pour accélérer les calculs.
- **Données iFixit :** Vérifiez régulièrement que les données de l'API iFixit sont mises à jour dans `guides.json`.

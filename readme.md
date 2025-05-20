# Chatbot RAG Support Technique

Un système de question-réponse intelligent basé sur l'architecture RAG (Retrieval-Augmented Generation) pour le support technique, exploitant des guides techniques et des discussions Reddit.

```
┌─────────────────────────────────────┐
│         INTERFACE WEB               │
│    (Flask + HTML/CSS + Tailwind)    │
└────────────────────┬────────────────┘
                     │
                     ▼
┌─────────────────────────────────────┐
│          MOTEUR RAG                 │
│  (LangChain + GPT-4.1 + Prompts)    │
└────────────────────┬────────────────┘
                     │
                     ▼
┌─────────────────────────────────────┐
│     SYSTÈME DE RÉCUPÉRATION         │
│  (FAISS + HuggingFace Embeddings)   │
└────────────────────┬────────────────┘
                     │
                    ┌┴┐
                    │ │
                    ▼ ▲
┌─────────────────────────────────────┐
│        SOURCES DE DONNÉES           │
│    (API Reddit PRAW + API iFixit)   │
└─────────────────────────────────────┘
```

## ✨ Caractéristiques

- ✅ **Interface intuitive** - Chat moderne avec animations et formatting Markdown
- 🔍 **Recherche intelligente** - Génération de requêtes alternatives pour une meilleure récupération
- 🧠 **Contexte enrichi** - Utilisation de posts Reddit et guides techniques avec récupération API dynamique
- 📊 **Transparence** - Visualisation des documents et requêtes utilisés pour chaque réponse
- 🚀 **Performance** - Optimisé avec CUDA pour les embeddings et FAISS pour la recherche vectorielle

## 📋 Prérequis

- Python 3.x
- Clé API OpenAI
- CUDA recommandé (mais non obligatoire)
- Fichiers sources JSON dans `./data/`

## 🚀 Installation

1. Cloner le dépôt
```bash
git clone https://github.com/CoCoRoRooo/TB
```

2. Créer et activer un environnement virtuel
```bash
# Sur Windows
.\venv\Scripts\activate
```

3. Installer les dépendances
```bash
pip install -r requirements.txt
```

4. Configurer la clé API
```bash
echo "OPENAI_KEY=votre_cle_api" > .env
```

5. Lancer l'application
```bash
python app.py
```

6. Ouvrir votre navigateur à l'adresse
```
http://localhost:5000
```

## 🧩 Architecture

Le système est composé de plusieurs modules interconnectés:

### Backend
- `app.py` - Application Flask principale et API REST
- `rag_chain.py` - Configuration des chaînes LangChain pour le RAG
- `retriever.py` - Gestion de l'indexation et de la recherche vectorielle
- `api_client.py` - Client pour l'API iFixit (récupération des étapes des guides)
- `utils.py` - Fonctions utilitaires diverses

### Frontend
- `templates/index.html` - Structure HTML de l'interface
- `static/css/style.css` - Styles CSS personnalisés
- `static/js/main.js` - Logique JavaScript pour les interactions utilisateur

### Scripts de collecte de données
- `get_posts.py` - Récupération des posts Reddit du subreddit "techsupport"
- `guides.py` - Récupération des guides techniques depuis l'API iFixit

## 💬 Utilisation

1. Saisissez votre question technique dans la zone de texte en bas de l'interface
2. Le système va:
   - Générer plusieurs reformulations de votre question pour améliorer la recherche
   - Récupérer les documents pertinents (posts Reddit et guides techniques)
   - Enrichir les guides avec leurs étapes détaillées via l'API
   - Générer une réponse structurée en se basant sur le contexte récupéré
3. Consultez les sections dépliables pour explorer:
   - Les requêtes alternatives générées
   - Les documents récupérés avec leurs métadonnées
   - Le temps de traitement pour la transparence

## 🔧 Technologies utilisées

- **Flask** - Serveur web et API REST
- **LangChain** - Orchestration des flux RAG
- **FAISS** - Indexation et recherche vectorielle efficace
- **Hugging Face** - Modèle d'embeddings `sentence-transformers/all-MiniLM-L6-v2`
- **OpenAI GPT-4.1** - Génération de requêtes et de réponses
- **Tailwind CSS** - Framework CSS pour l'interface utilisateur
- **PRAW** - API Reddit pour la collecte de données

## 📝 Format des réponses

Les réponses générées suivent une structure claire:

- **🔍 Analyse du problème** - Synthèse et compréhension de la question
- **✅ Vérifications préalables** - Étapes de diagnostic recommandées
- **📝 Procédure détaillée** - Instructions étape par étape pour résoudre le problème
- **💡 Conseils supplémentaires** - Recommandations additionnelles et bonnes pratiques
- **🔗 Sources consultées** - Références aux documents utilisés pour générer la réponse

## 🛠️ Personnalisation

### Modèles
- Pour changer le modèle d'embeddings, modifiez le paramètre `model_name` dans `retriever.py`
- Pour changer le modèle LLM, modifiez la configuration dans `rag_chain.py`

### Sources de données
- Ajoutez de nouvelles sources en modifiant les fonctions de chargement dans `retriever.py`
- Adaptez le format des documents dans la fonction `index_data_embeddings()`

### Interface utilisateur
- Personnalisez le design en modifiant `templates/index.html` et `static/css/style.css`
- Ajoutez de nouvelles fonctionnalités en étendant `static/js/main.js`

## 📊 Performance et optimisation

- Utilisation de CUDA pour accélérer les calculs d'embeddings si disponible
- Optimisation du retriever avec des paramètres ajustables:
  - `chunk_size` et `chunk_overlap` pour le découpage des documents
  - `search_type` et `search_kwargs` pour la configuration de la recherche
- Affichage du temps de traitement pour chaque requête

## 🙏 Remerciements

- [iFixit](https://www.ifixit.com) pour leur API publique de guides techniques
- [Reddit](https://www.reddit.com) et la communauté r/techsupport pour les discussions techniques
- [LangChain](https://github.com/langchain-ai/langchain) pour le framework RAG
- [HuggingFace](https://huggingface.co) pour les modèles d'embeddings
- [OpenAI](https://openai.com) pour les modèles de langage

# 🚀 Assistant Technique avec IA et Recherche de Guides

Ce projet est une **API Flask** avec une **interface web** qui permet aux utilisateurs de poser des questions techniques. Le système utilise :

- **FAISS** pour la recherche vectorielle des guides les plus pertinents
- **OpenAI GPT-4** pour générer des réponses détaillées
- **Flask** pour gérer l'API et la communication
- **Bootstrap & CSS** pour une interface utilisateur moderne et responsive

---

## 📁 Structure des fichiers

```
📂 projet-assistant-technique
│-- 📂 static/               # Fichiers CSS pour le style
│   ├── styles.css          # Feuille de style principale
│-- 📂 templates/            # Fichiers HTML
│   ├── index.html          # Interface utilisateur
│-- 📂 data/                 # Fichiers de données
│   ├── guides.json         # Base de connaissances (guides techniques)
│-- .env                    # Stocke la clé API OpenAI
│-- app.py                  # API Flask
│-- tokenizer.py            # Gestion de l'indexation FAISS
│-- llm.py                  # Génération de réponses via GPT-4
│-- README.md               # Documentation du projet
```

---

## 🛠️ Installation et exécution

### 1️⃣ Prérequis

- **Python 3.8+**
- **Clé API OpenAI** (à ajouter dans `.env`)

### 2️⃣ Installation des dépendances

```bash
pip install -r requirements.txt
```

#### Explication des dépendances :

- **flask** → Pour créer l'API backend qui communique avec l'interface web.
- **faiss-cpu** → Pour la recherche rapide des guides en utilisant des embeddings.
- **sentence-transformers** → Pour générer les embeddings des textes avec _all-MiniLM-L6-v2_.
- **openai** → Pour interagir avec l'API OpenAI et générer des réponses via ChatGPT.
- **python-dotenv** → Pour charger la clé API OpenAI depuis un fichier .env de manière sécurisée.

### 3️⃣ Lancement du serveur Flask

```bash
python app.py
```

Le serveur tournera sur `http://127.0.0.1:5000/`

---

## 📜 Explication des fichiers

### **1️⃣ `tokenizer.py`** (Indexation des guides avec FAISS)

Ce fichier :

- Charge un fichier JSON contenant des guides techniques
- Convertit les guides en **vecteurs** avec `SentenceTransformer`
- Crée une **base FAISS** pour la recherche rapide

### **2️⃣ `llm.py`** (Génération de réponses avec GPT-4)

Ce fichier :

- Charge les guides indexés
- Trouve les guides les plus pertinents via FAISS
- Génère une réponse avec **OpenAI GPT-4**

### **3️⃣ `app.py`** (API Flask)

Ce fichier :

- Expose une API `POST /ask` qui prend une question et renvoie une réponse
- Charge FAISS et le modèle de génération

```python
@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    query = data.get("query")
    top_guides = search_guides(query, faiss_index, guide_texts, embed_model)
    response = generate_response(query, top_guides)
    return jsonify({"response": response})
```

### **4️⃣ `index.html`** (Interface utilisateur)

Une page HTML stylisée qui permet de poser des questions via un champ de saisie et un bouton.

### **5️⃣ `styles.css`** (Style de la page)

Ajoute des styles modernes avec **Bootstrap** et du CSS personnalisé.

---

## 🔥 Exemples d'utilisation

1️⃣ **Poser une question**

```json
POST /ask
{
    "query": "Mon iPhone ne charge plus"
}
```

2️⃣ **Réponse du chatbot**

```json
{
  "response": "Essayez d'utiliser un autre câble et vérifiez le port Lightning."
}
```

---

## 🎯 Améliorations possibles

- ✅ Supporter d'autres modèles d'IA (Mistral, Llama...)
- ✅ Ajouter un système d'historique des questions
- ✅ Interface utilisateur encore plus interactive

🚀 **Projet clé en main pour un assistant technique intelligent !**

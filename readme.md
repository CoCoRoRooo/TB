# 🚀 Assistant Technique avec IA et Recherche de Guides

Ce projet est une **API Flask** avec une **interface web** qui permet aux utilisateurs de poser des questions techniques. Le système utilise :

- **FAISS** pour la recherche vectorielle des guides les plus pertinents
- **Mistral (via Ollama) ou OpenAI GPT-4** pour générer des réponses intelligentes
- **Flask** pour gérer l'API et la communication
- **LangChain** pour orchestrer la chaîne de traitement RAG

---

## 📁 Structure des fichiers

```
📂 TB
│-- 📂 static/               # Fichiers CSS pour le style
│   ├── styles.css          # Feuille de style principale
│-- 📂 templates/            # Fichiers HTML
│   ├── index.html          # Interface utilisateur
│-- 📂 data/                 # Fichiers de données
│   ├── guides.json         # Base de connaissances (guides techniques)
│   ├── guides.py           # Récupération et stockage des guides depuis iFixIt
│-- .env                    # Stocke la clé API OpenAI
│-- app.py                  # API Flask
│-- tokenizer.py            # Gestion de l'indexation FAISS
│-- llm.py                  # Génération de réponses avec Mistral/OpenAI
│-- README.md               # Documentation du projet
```

---

## 🛠️ Installation et exécution

### 1️⃣ Prérequis

- **Python 3.8+**
- **Clé API OpenAI** (si utilisation de GPT-4, à ajouter dans `.env`)
- **Ollama** installé pour utiliser Mistral en local

### 2️⃣ Installation des dépendances

```bash
pip install -r requirements.txt
```

#### Explication des dépendances :

- **flask** → Pour créer l'API backend qui communique avec l'interface web.
- **python-dotenv** → Pour charger des variables d'environnement depuis un fichier .env de manière sécurisée.
- **langchain** → Pour orchestrer la chaîne RAG avec retrieval et LLM.
- **ollama** → Pour exécuter Mistral localement.
- **langchain_community** → Module complémentaire pour LangChain.
- **langchain_openai** → Pour interagir avec l'API OpenAI et générer des réponses via ChatGPT.

### 3️⃣ Lancement du serveur Flask

```bash
python app.py
```

Le serveur tournera sur `http://127.0.0.1:5000/`

---

## 📝 Explication des fichiers

### **1️⃣ `tokenizer.py`** (Indexation des guides avec FAISS)

Ce fichier :

- Charge un fichier JSON contenant des guides techniques
- Convertit les guides en **vecteurs** avec `HuggingFaceEmbeddings`
- Crée une **base FAISS** pour la recherche rapide

Extrait de code :

```python
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L6-v2")
vector_store = FAISS.from_documents(documents, embedding_model)
retriever = vector_store.as_retriever()
```

---

### **2️⃣ `llm.py`** (Génération de réponses avec GPT-4 ou Mistral via Ollama)

Ce fichier :

- Charge les guides indexés
- Trouve les guides les plus pertinents via FAISS
- Génère une réponse avec **GPT-4 ou Mistral**
- Utilise **LangChain** pour structurer le processus

Extrait de code :

```python
class OllamaLLM(LLM):
    model: str = "mistral"

    def _call(self, prompt: str, stop: List[str] = None) -> str:
        response = ollama.chat(model=self.model, messages=[{"role": "user", "content": prompt}])
        return response["message"]["content"]
```

On peut choisir le modèle utilisé en modifiant cette ligne :

```python
use_ollama = True  # Mettre à False pour utiliser OpenAI
```

---

### **3️⃣ `app.py`** (API Flask)

Ce fichier :

- Expose une API `POST /ask` qui prend une question et renvoie une réponse
- Charge FAISS et le modèle de génération

Extrait de code :

```python
@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    user_query = data.get("query", "")

    if not user_query:
        return jsonify({"error": "Aucune question fournie"}), 400

    response = rag_chain.invoke(user_query)
    return jsonify({"query": user_query, "response": response})
```

---

### **4️⃣ `index.html`** (Interface utilisateur)

Une page HTML stylisée qui permet de poser des questions via un champ de saisie et un bouton.

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

- ✅ Supporter d'autres modèles d'IA (Llama, Claude...)
- ✅ Ajouter un système d'historique des questions
- ✅ Interface utilisateur encore plus interactive

🚀 **Projet clé en main pour un assistant technique intelligent !**

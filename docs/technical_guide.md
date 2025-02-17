# 📖 Documentation Technique du Projet

## 📌 Introduction

Ce projet est une application Flask qui permet aux utilisateurs de poser des questions et d'obtenir des réponses basées sur un ensemble de guides techniques extraits de l'API iFixit. L'application utilise le traitement du langage naturel et la recherche vectorielle pour proposer des réponses pertinentes.

---

## 📂 Structure des fichiers

- **`data/guides.json`** : Contient les guides techniques extraits de l'API iFixit.
- **`data/guides.py`** : Script permettant de récupérer les guides depuis l'API iFixit et de les stocker localement.
- **`tokenizer.py`** : Permet de convertir les guides en vecteurs et d'effectuer des recherches efficaces.
- **`llm.py`** : Interagit avec l'API OpenAI pour générer des réponses aux questions des utilisateurs.
- **`app.py`** : Contient l'application Flask qui expose les endpoints pour l'interaction utilisateur.
- **`templates/index.html`** : Interface utilisateur pour poser des questions.

---

## 📝 Détails des fichiers

### 📁 `data/guides.json`

#### 📄 Contenu

Fichier JSON contenant une liste de guides techniques. Chaque guide est représenté sous la forme d'un objet avec les champs suivants :

- **`guideid`** : Identifiant unique du guide.
- **`locale`** : Langue du guide.
- **`title`** : Titre du guide.
- **`summary`** : Brève description.
- **`difficulty`** : Niveau de difficulté.
- **`url`** : Lien vers le guide sur iFixit.
- **`image`** : Liens vers différentes versions d'images associées.

---

### 📁 `data/guides.py`

#### 🔹 Fonction `fetch_all_guides()`

Récupère tous les guides disponibles via l'API iFixit et les stocke dans `guides.json`.

- Utilise la pagination avec `offset` et `limit`.
- Gère les erreurs JSON et les limitations de requêtes API.
- Stocke les guides sous format JSON.

```python
url = f"https://www.ifixit.com/api/2.0/guides?offset={offset}&limit={limit}"
response = requests.get(url)
if response.status_code == 200:
    guides = response.json()
    all_guides.extend(guides)
```

---

### 📁 `tokenizer.py`

#### 🔹 Fonction `load_guides(file_path)`

Charge les guides depuis le fichier JSON pour traitement.

#### 🔹 Fonction `index_guides(guides, model_name="all-MiniLM-L6-v2")`

Crée des embeddings pour chaque guide et stocke ces vecteurs dans FAISS.

- Utilise le modèle `SentenceTransformer`.
- Convertit les guides en texte.
- Crée un index FAISS basé sur la distance euclidienne.

```python
model = SentenceTransformer(model_name)
embeddings = model.encode(texts, convert_to_numpy=True)
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)
```

#### 🔹 Fonction `search_guides(query, faiss_index, guide_texts, model, top_k=3)`

Effectue une recherche dans FAISS pour trouver les guides les plus pertinents.

```python
query_embedding = model.encode([query], convert_to_numpy=True)
distances, indices = faiss_index.search(query_embedding, top_k)
results = [guide_texts[i] for i in indices[0]]
```

---

### 📁 `llm.py`

#### 🔹 Fonction `generate_response(user_query, relevant_guides)`

Génère une réponse basée sur les guides trouvés à l'aide de l'API OpenAI.

- Formate la question et les guides en contexte.
- Envoie une requête à OpenAI GPT-4.
- Renvoie la meilleure réponse générée.

```python
response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "Tu es un assistant technique."},
        {"role": "user", "content": prompt},
    ],
    temperature=0.5,
)
```

---

### 📁 `app.py`

#### 🔹 Fonction `ask()`

Endpoint API permettant de poser une question et d'obtenir une réponse.

- Vérifie la requête.
- Recherche les guides pertinents via `search_guides()`.
- Génère une réponse avec `generate_response()`.
- Retourne la réponse sous format JSON.

```python
@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    user_query = data.get("query", "")
    top_guides = search_guides(user_query, faiss_index, guide_texts, embed_model)
    response = generate_response(user_query, top_guides)
    return jsonify({"query": user_query, "guides": top_guides, "response": response})
```

---

### 📁 `templates/index.html`

Interface utilisateur simple permettant de poser des questions.

- Champ de saisie utilisateur.
- Bouton d'envoi.
- Affichage de la réponse reçue.

```html
<input
  type="text"
  id="questionInput"
  class="form-control"
  placeholder="Ex: Mon iPhone ne charge plus"
/>
<button class="btn btn-primary w-100" onclick="askQuestion()">
  Poser la question
</button>
```

---

## 🚀 Conclusion

Ce projet permet d'exploiter des guides techniques via une recherche optimisée et une intelligence artificielle générative. L'intégration de FAISS pour la recherche vectorielle et de GPT-4 pour la génération de réponses offre un assistant technique performant et efficace.

# 📖 Documentation Technique du Projet

## 📌 Introduction

Ce projet est une application Flask qui permet aux utilisateurs de poser des questions et d'obtenir des réponses basées sur un ensemble de guides techniques extraits de l'API iFixit. L'application utilise le traitement du langage naturel et la recherche vectorielle pour proposer des réponses pertinentes.

---

## 📂 Structure des fichiers

- **`data/guides.json`** : Contient les guides techniques extraits de l'API iFixit.
- **`data/guides.py`** : Script permettant de récupérer les guides depuis l'API iFixit et de les stocker localement.
- **`tokenizer.py`** : Permet de convertir les guides en vecteurs et d'effectuer des recherches efficaces.
- **`llm.py`** : Interagit avec l'API OpenAI/Ollama pour générer des réponses aux questions des utilisateurs.
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

Cette fonction permet de récupérer tous les guides disponibles via l'API iFixit. Elle gère la pagination avec les paramètres `offset` et `limit`, et traite les erreurs JSON et les limitations de requêtes API.

- Utilise `requests.get` pour interroger l'API d'iFixit.
- Si la requête est réussie, elle étend la liste des guides récupérés avec `all_guides`.
- En cas de limitation de requêtes `(status code 429)`, elle met en pause - l'exécution pour 30 secondes avant de réessayer.

```python
def fetch_all_guides():
    all_guides = []
    offset = 0
    limit = 20
    has_more = True

    while has_more:
        url = f"https://www.ifixit.com/api/2.0/guides?offset={offset}&limit={limit}"
        response = requests.get(url)

        # Vérifiez la réponse
        print(f"Status Code: {response.status_code}")
        print(
            f"Response Content: {response.content[:200]}..."
        )  # Affiche les premiers 200 caractères de la réponse

        if response.status_code == 200:
            try:
                guides = response.json()
                all_guides.extend(guides)

                if len(guides) < limit:
                    has_more = False
                else:
                    offset += limit
            except ValueError:
                print("Erreur de décodage JSON")
                break
        elif response.status_code == 429:
            # Si la limite est atteinte, attendons 30 secondes avant de continuer
            print("Limite de requêtes atteinte. Attente de 30 secondes...")
            time.sleep(30)  # Attendre 30 secondes avant de réessayer
        else:
            print(
                f"Erreur lors de la récupération des données. Code statut: {response.status_code}"
            )
            break

    return all_guides
```

---

### 📁 `tokenizer.py`

#### 🔹 Fonction `load_guides(file_path)`

Charge les guides depuis le fichier JSON pour les transformer en vecteurs. Ce fichier est utilisé pour initialiser les données en amont du traitement par LangChain.

#### 🔹 Fonction `index_guides(guides, model_name="all-MiniLM-L6-v2")`

Cette fonction est responsable de la création des embeddings pour chaque guide. Elle utilise LangChain pour créer des objets `Document` et les indexer dans une base FAISS (Fast Approximate Nearest Neighbor Search). FAISS permet d'effectuer des recherches rapides dans les vecteurs d'embeddings.

- Les guides sont convertis en objets `Document` compatibles avec LangChain.
- Utilisation du modèle `SentenceTransformer` (`ici all-MiniLM-L6-v2`) pour générer des embeddings des guides.
- La base FAISS permet ensuite une recherche rapide et efficace dans ces embeddings.
- La fonction retourne un `Retriever` LangChain, utilisé pour interroger les documents indexés.

```python
# Convertir les guides en vecteurs et créer un retriever LangChain
def index_guides(guides, model_name="sentence-transformers/paraphrase-MiniLM-L6-v2"):
    # Construire les textes et les objets Document
    documents = [
        Document(
            page_content=f"{g['dataType']} - {g['type']} {g['subject']} : {g['title']} {(g['url'])}",
            metadata={
                "url": g["url"],
                "type": g["type"],
                "subject": g["subject"],
                "title": g["title"],
            },
        )
        for g in guides
    ]

    # Créer des embeddings avec LangChain
    embedding_model = HuggingFaceEmbeddings(model_name=model_name)
    vector_store = FAISS.from_documents(documents, embedding_model)

    return vector_store.as_retriever(), embedding_model
```

### 📁 `llm.py`

#### 🔹 Classe `OllamaLLM`

Cette classe permet d'utiliser un modèle local Ollama pour générer des réponses en fonction des prompts. Elle repose sur la librairie `ollama` pour communiquer avec un modèle de traitement de langage.

```python
class OllamaLLM(LLM):
    model: str = "mistral"

    def _call(self, prompt: str, stop: List[str] = None) -> str:
        response = ollama.chat(
            model=self.model, messages=[{"role": "user", "content": prompt}]
        )
        return response["message"]["content"]

    @property
    def _identifying_params(self) -> dict:
        return {"model": self.model}

    @property
    def _llm_type(self) -> str:
        return "ollama"
```

#### 🔹 Fonction `create_rag_chain(retriever)`

Cette fonction crée une chaîne RAG (Retrieval-Augmented Generation). Elle récupère des documents pertinents via le `retriever` LangChain et génère une réponse en utilisant un modèle LLM, comme Ollama ou OpenAI.

```python
# Chaîne RAG
def create_rag_chain(retriever):
    return (
        {
            "context": retriever | format_documents,
            "question": RunnablePassthrough(),
        }
        | prompt_template
        | llm
        | StrOutputParser()
    )
```

##### Fonctionnement de `create_rag_chain(retriever)`

La fonction est utilisée pour configurer une chaîne de traitement qui :

- Récupère des documents pertinents via un mécanisme de récupération d'information (le `retriever` LangChain).
- Formate ces documents pour les rendre compatibles avec un modèle de génération de texte (par exemple, Ollama ou OpenAI).
- Applique un prompt template pour construire un contexte clair et structuré pour le modèle de génération de texte.
- Génère une réponse en utilisant un modèle de langage (LLM, par exemple Ollama ou OpenAI).
- Parse la sortie pour en faire une réponse structurée.

##### Étape par étape

1. `"context": retriever | format_documents`

   - But : Cette partie de la chaîne est responsable de la récupération des documents pertinents pour la question de l'utilisateur.
   - Explication détaillée :
     - `retriever` est un objet LangChain, généralement un `Retriever` basé sur un moteur de recherche vectorielle comme FAISS. Il est utilisé pour récupérer les documents les plus pertinents par rapport à une question posée.
     - Ensuite, `| format_documents` est une opération qui prend les documents récupérés et les transforme en un format lisible par le modèle de langage. `format_documents` crée une chaîne de texte en concaténant les contenus des documents avec des séparateurs appropriés. Cette étape permet au modèle LLM de comprendre facilement les informations extraites.
   - Importance : Sans cette étape, le modèle de génération de texte ne disposerait d'aucun contexte pertinent sur lequel se baser pour générer une réponse.

2. `"question": RunnablePassthrough()`

   - But : Cette partie passe la question de l'utilisateur telle quelle à l'étape suivante sans la modifier.
   - Explication détaillée :
     - `RunnablePassthrough` est un composant de LangChain qui prend un objet ou une valeur en entrée et la transmet telle quelle en sortie. Ici, il est utilisé pour s'assurer que la question de l'utilisateur est transmise sans modification à l'étape suivante.
   - Importance : Il est crucial que la question de l'utilisateur soit intacte pour qu'elle puisse être intégrée dans le prompt template pour la génération de texte.

3. `| prompt_template`

   - But : Appliquer un modèle de prompt sur le contexte et la question pour formater correctement la demande avant de la transmettre au modèle LLM.
   - Explication détaillée :

     - `prompt_template` est un objet LangChain de type `PromptTemplate` qui définit la structure du prompt. Il utilise deux variables : `context` (les documents récupérés) et `question` (la question posée par l'utilisateur). Le prompt template construit un message structuré qui sera envoyé au modèle LLM pour générer une réponse.
     - Exemple de prompt :

       ```python
        L'utilisateur pose la question : {question}

        Voici des guides pertinents :
        {context}

        Réponds en t'appuyant sur ces guides.
       ```

   - Importance : Cette étape est essentielle pour fournir au modèle LLM un format de question et de contexte structuré et facile à traiter, ce qui augmente la pertinence et la qualité de la réponse générée.

4. `| llm`

   - But : Cette étape utilise un modèle de langage (LLM) pour générer une réponse à partir du prompt formaté.
   - Explication détaillée :
     - Le modèle LLM (par exemple, Ollama ou OpenAI) prend le prompt formaté et génère une réponse basée sur son entraînement et les informations fournies dans le contexte.
   - Importance : Le LLM est le cœur du processus de génération de la réponse. Sans lui, il n'y aurait aucune génération automatique de texte. C'est lui qui utilise les informations extraites des guides pour fournir une réponse cohérente et pertinente à l'utilisateur.

5. `| StrOutputParser()`

   - But : Cette étape sert à parser et à formater la sortie générée par le modèle LLM, afin de la rendre propre et structurée.
   - Explication détaillée :
     - `StrOutputParser()` est un composant de LangChain qui prend la sortie du modèle de langage et la transforme en une chaîne de caractères lisible et propre.
   - Importance : Cette étape permet de nettoyer la sortie générée par le modèle pour s'assurer que la réponse est claire et bien formatée avant de l'envoyer à l'utilisateur.

##### Ordre d'appel et syntaxe

- Ordre d'exécution : La syntaxe de LangChain utilise le "pipe" (`|`) pour enchaîner les étapes, ce qui signifie que chaque composant de la chaîne reçoit la sortie du composant précédent comme entrée.
- Flux d'exécution :
  1. Les documents sont récupérés et formatés.
  2. La question de l'utilisateur est transmise sans modification.
  3. Un prompt structuré est créé avec le contexte et la question.
  4. Le modèle LLM génère une réponse en fonction du prompt.
  5. La réponse générée est ensuite nettoyée et renvoyée sous forme de texte lisible.

##### Importance de chaque étape

1. Récupération des documents (`retriever`) : Sans la recherche d'informations pertinentes, le modèle ne pourrait pas répondre de manière appropriée. Cette étape garantit que le modèle reçoit des données fiables et ciblées.

2. Formatage des documents (`format_documents`) : La conversion des documents en un format compréhensible pour le modèle est cruciale pour assurer que le LLM peut extraire les informations pertinentes et générer une réponse cohérente.

3. Construction du prompt : Un bon prompt est essentiel pour guider le modèle vers une réponse correcte et précise. Un prompt mal conçu peut entraîner des réponses vagues ou inexactes.

4. Modèle LLM : Le modèle LLM transforme le prompt structuré en une réponse textuelle. Sa qualité et sa capacité à comprendre le contexte sont cruciales pour obtenir une réponse utile.

5. Parsing de la sortie : Une sortie propre et bien formatée est essentielle pour garantir que l'utilisateur reçoit une réponse lisible et claire.

#### 🔹 Définition du prompt

Le `PromptTemplate` définit comment la question et les documents pertinents doivent être présentés au modèle LLM pour générer une réponse.

```python
# Définir le prompt
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="L'utilisateur pose la question : {question}\n\nVoici des guides pertinents :\n{context}\n\nRéponds en t'appuyant sur ces guides.",
)
```

#### 🔹 Fonction `format_documents(docs)`

Cette fonction extrait le contenu des documents trouvés par le retriever et le prépare pour l'input au modèle.

```python
# Fonction qui récupère le contenu des documents trouvés
def format_documents(docs):
    return "\n\n".join(doc.page_content for doc in docs)
```

---

### 📁 `app.py`

#### 🔹 Charger et indexer les guides au démarrage du serveur

Au démarrage du serveur Flask, cette ligne charge et indexe les guides depuis `guides.json`, en utilisant les fonctions définies dans `tokenizer.py`.

```python
guides = load_guides("data/guides.json")
retriever, embed_model = index_guides(guides)
```

#### 🔹 Construire la chaîne RAG

Cette ligne crée la chaîne RAG avec le retriever LangChain.

```python
rag_chain = create_rag_chain(retriever)
```

#### 🔹 Endpoint `/`

Affiche la page d'accueil avec un champ pour poser des questions.

```python
def home():
    return render_template("index.html")
```

#### 🔹 Endpoint `/ask`

- Récupère la requête utilisateur.
- Recherche les guides pertinents via le retriever LangChain.
- Génère une réponse avec la chaîne RAG.

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

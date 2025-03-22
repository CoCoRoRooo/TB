from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models import LLM
from typing import Any, List
import ollama
import os
import requests
from dotenv import load_dotenv


class OllamaLLM(LLM):
    model: str = "llama3"  # mistral, llama3

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


# Charger les variables d'environnement
load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_KEY")

# Utiliser un modèle OpenAI ou Ollama
use_ollama = False  # Mettre à False pour utiliser OpenAI

if not use_ollama:
    llm = ChatOpenAI(openai_api_key=OPENAI_KEY, model="gpt-4", temperature=0.5)
else:
    llm = OllamaLLM()

prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
L'utilisateur pose la question suivante :

➡️ {question}

Tu disposes ci-dessous d'une sélection de guides techniques pertinents. Ces guides incluent des descriptions générales ainsi que des **étapes de résolution (Steps)** détaillées.

🎯 Ta mission :
- Exploite au maximum les **étapes (Steps)** fournies dans les guides pour construire ta réponse.
- Fournis une réponse claire, structurée et complète en français.
- Formule des instructions pratiques en t'appuyant précisément sur les étapes, tout en adaptant ton discours au contexte du problème posé.
- Donne la source du guide.

📚 Guides disponibles :

{context}

🛠 Format de réponse attendu :
---
🔍 **Analyse du problème** :
[Présente une synthèse du problème posé et ce que les guides indiquent à son sujet.]

✅ **Vérifications préalables recommandées** :
[Liste les éléments à inspecter ou tester avant de démarrer la procédure.]

📝 **Procédure détaillée (basée sur les Steps)** :
Step 1 : [Décris l'étape avec clarté, en reformulant si nécessaire pour être pédagogique]  
Step 2 : [...]  
Step 3 : [...]  
...

💡 **Conseils supplémentaires ou précautions à prendre** :
[Ajoute ici des recommandations pratiques, conseils d’outillage, précautions de sécurité, ou bonnes pratiques.]

---

🎯 Important : Structure ta réponse de manière fluide et rigoureuse. Sois synthétique mais informatif, professionnel dans le ton, et précis dans les formulations.
""",
)


def get_guide_steps(guideid):
    url = f"https://www.ifixit.com/api/2.0/guides/{guideid}"
    response = requests.get(url)

    if response.status_code != 200:
        return {
            "error": f"Échec de récupération du guide {guideid}, code: {response.status_code}"
        }

    data = response.json()
    steps = []

    cpt_steps = 0

    for step in data.get("steps", []):
        cpt_steps += 1
        step_texts = [
            line["text_rendered"]
            for line in step.get("lines", [])
            if "text_rendered" in line
        ]
        steps.append({"stepno": cpt_steps, "text": step_texts})

    return steps


# Fonction qui récupère le contenu des documents trouvés
def format_documents(docs):
    for doc in docs:
        guide_steps = get_guide_steps(doc.metadata.get("guideid"))

        doc.page_content += "\n".join(
            f"Step {step['stepno']}:\n" + "\n".join(step["text"])
            for step in guide_steps
        )

    print(docs)

    return "\n\n".join(doc.page_content for doc in docs)


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

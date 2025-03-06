from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models import LLM
from typing import Any, List
import ollama
import os
from dotenv import load_dotenv


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


# Charger les variables d'environnement
load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_KEY")

# Utiliser un modèle OpenAI ou Ollama
use_ollama = True  # Mettre à False pour utiliser OpenAI

if not use_ollama:
    llm = ChatOpenAI(openai_api_key=OPENAI_KEY, model="gpt-4", temperature=0.5)
else:
    llm = OllamaLLM()

# Définir le prompt
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="L'utilisateur pose la question : {question}\n\nVoici des guides pertinents :\n{context}\n\nRéponds en t'appuyant sur ces guides.",
)


# Fonction qui récupère le contenu des documents trouvés
def format_documents(docs):
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

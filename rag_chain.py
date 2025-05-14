import logging
import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

from retriever import load_guides, load_posts, index_data_embeddings
from utils import format_documents, get_unique_union

logger = logging.getLogger(__name__)

# Variables globales pour stocker les chaînes de traitement
_retriever = None
_rag_chain = None
_generate_queries = None
_retrieval_chain = None


def get_retriever():
    """Retourne l'instance du retriever"""
    global _retriever
    return _retriever


def get_rag_chain():
    """Retourne l'instance de la chaîne RAG"""
    global _rag_chain
    return _rag_chain


def get_generate_queries():
    """Retourne l'instance de la chaîne de génération de requêtes"""
    global _generate_queries
    return _generate_queries


def get_retrieval_chain():
    """Retourne l'instance de la chaîne de récupération"""
    global _retrieval_chain
    return _retrieval_chain


def initialize_rag_system():
    """Initialise le système RAG avec les chaînes de traitement"""
    global _retriever, _rag_chain, _generate_queries, _retrieval_chain

    logger.info("Initialisation du système RAG")

    # Charger les variables d'environnement
    load_dotenv()
    OPENAI_KEY = os.getenv("OPENAI_KEY")

    # Charger les données
    posts = load_posts("./data/techsupport_posts.json")
    guides = load_guides("./data/guides.json")

    # Créer le retriever
    _retriever = index_data_embeddings(posts, guides)

    # Le prompt pour le modèle
    final_prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""
L'utilisateur pose la question suivante :

➡️ {question}

Tu disposes ci-dessous de guides techniques et de posts Reddit pertinents. Ces contenus incluent des descriptions générales, des conseils pratiques, des solutions proposées par la communauté, et parfois des instructions techniques détaillées.

🎯 Ta mission :

- Analyse et synthétise les informations issues des guides techniques et des posts Reddit pour répondre à la question.

- Fournis une réponse structurée et complète.

- Utilise les étapes décrites dans les guides techniques, si présentes, et les solutions suggérées par les utilisateurs dans les posts Reddit.

- Répond en Français

📚 Sources disponibles :

{context}

🛠 Format de réponse attendu :

---

🔍 Analyse du problème :

[Présente une synthèse du problème posé, en te basant sur les informations extraites des documents.]

✅ Vérifications préalables recommandées :

[Liste les éléments à inspecter ou tester avant de commencer les manipulations.]

📝 Procédure détaillée proposée :

[Utilise les étapes comme "Step 1", "Step 2" pour les guides iFixit, ou les conseils donnés dans les posts Reddit.]

💡 Conseils supplémentaires ou précautions à prendre :

[Ajoute des conseils supplémentaires tirés des guides ou des commentaires des utilisateurs.]

🔗 Sources consultées :

[Indique ici les URL des documents (guides ou posts Reddit) ayant servi à construire ta réponse. Utilise les URLs disponibles dans les métadonnées des documents fournis.]

---

🎯 Important : Structure ta réponse de manière fluide, concise, et professionnelle. Mentionne les sources utilisées, telles que l'URL du guide ou du post Reddit.
""",
    )

    # Prompt pour générer des requêtes alternatives
    prompt_template = PromptTemplate(
        input_variables=["question"],
        template="""
Ta tâche est de générer cinq reformulations différentes de la question posée par l'utilisateur afin de retrouver des documents pertinents dans une base de données vectorielle.

En proposant plusieurs perspectives sur la question, ton objectif est d'aider l'utilisateur à surmonter certaines limites de la recherche par similarité basée sur la distance.

Fournis ces questions alternatives, chacune séparée par un saut de ligne.

Répond en Anglais

Question initiale : {question}
""",
    )

    # Initialiser le LLM
    llm = ChatOpenAI(openai_api_key=OPENAI_KEY, model="gpt-4.1", temperature=0.5)

    # Chaîne de génération de requêtes
    _generate_queries = (
        prompt_template | llm | StrOutputParser() | (lambda x: x.split("\n"))
    )

    # Chaîne de récupération
    _retrieval_chain = _generate_queries | _retriever.map() | get_unique_union

    # Chaîne RAG complète
    _rag_chain = (
        {
            "context": _retrieval_chain,
            "question": RunnablePassthrough(),
        }
        | final_prompt_template
        | llm
        | StrOutputParser()
    )

    logger.info("Système RAG initialisé avec succès")

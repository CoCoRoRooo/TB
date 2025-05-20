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


def debug_context(context):
    print("=== CONTENU DU CONTEXT ===")
    print(context)
    print("==========================")
    return context


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

Tu disposes uniquement des documents suivants : des guides techniques et des posts Reddit pertinents.
Ces contenus incluent des descriptions générales, des conseils pratiques, des solutions proposées par la communauté, et parfois des instructions techniques détaillées.

🎯 Ta mission :

    Base strictement ta réponse sur les informations présentes dans les documents fournis ci-dessous ({context}).

    N'utilise aucune connaissance extérieure. Si une information n’est pas présente, indique-le explicitement.

    Fournis une réponse structurée, professionnelle et en français.

📚 Sources disponibles :

{context}

🛠 Format de réponse attendu :

🔍 Analyse du problème :

[Présente une synthèse du problème posé, uniquement en te basant sur les documents.]

✅ Vérifications préalables recommandées :

[Liste les éléments à inspecter ou tester avant toute manipulation, tels que suggérés dans les documents.]

📝 Procédure détaillée proposée :

[Structure la procédure étape par étape : “Étape 1”, “Étape 2”… en t’appuyant sur les guides ou les conseils Reddit.]

💡 Conseils ou précautions à prendre :

[Ajoute ici uniquement les recommandations explicitement mentionnées dans les documents.]

🔗 Sources consultées :

[Liste les URL des documents (guides ou posts Reddit) utilisés pour construire la réponse, selon les métadonnées disponibles.]

📌 Important :
Tu dois strictement t'appuyer sur les contenus fournis dans {context}.
Aucune inférence ou ajout personnel n’est autorisé. Si la réponse n’est pas déductible des documents, indique-le clairement.
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
    llm = ChatOpenAI(openai_api_key=OPENAI_KEY, model="gpt-4.1", temperature=0.0)

    # Chaîne de génération de requêtes
    _generate_queries = (
        prompt_template | llm | StrOutputParser() | (lambda x: x.split("\n"))
    )

    # Chaîne de récupération
    _retrieval_chain = _generate_queries | _retriever.map() | get_unique_union

    # Chaîne RAG complète
    _rag_chain = (
        {
            "context": _retrieval_chain
            | format_documents
            | (lambda x: debug_context(x)),
            "question": RunnablePassthrough(),
        }
        | final_prompt_template
        | llm
        | StrOutputParser()
    )

    logger.info("Système RAG initialisé avec succès")

import logging
import json
import time
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

logger = logging.getLogger(__name__)


def load_guides(file_path):
    """Charge les guides depuis un fichier JSON"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            guides = json.load(f)
        logger.info(f"Guides chargés avec succès: {len(guides)} guides trouvés")
        return guides
    except Exception as e:
        logger.error(f"Erreur lors du chargement des guides: {e}")
        return []


def load_posts(file_path):
    """Charge les posts depuis un fichier JSON"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            posts = json.load(f)
        logger.info(f"Posts chargés avec succès: {len(posts)} posts trouvés")
        return posts
    except Exception as e:
        logger.error(f"Erreur lors du chargement des posts: {e}")
        return []


def index_data_embeddings(
    posts, guides, model_name="sentence-transformers/all-MiniLM-L6-v2"
):
    """Convertit les posts et guides en vecteurs et crée un retriever LangChain"""
    logger.info("Début de l'indexation des données")
    start_time = time.time()

    # Construire les textes et les objets Document pour les posts
    documents = []
    for p in posts:
        text_comments = ""
        for comment in p.get("comments", []):
            text_comments += comment + "\n"
        documents.append(
            Document(
                page_content=f"{p['titre']} - {p['contenu']}",
                metadata={
                    "comments": text_comments,
                    "url": p.get("url", ""),
                    "titre": p.get("titre", ""),
                    "contenu": p.get("contenu", ""),
                },
            )
        )

    # Construire les textes et les objets Document pour les guides
    for g in guides:
        documents.append(
            Document(
                page_content=f"{g.get('dataType', '')} - {g.get('type', '')} {g.get('subject', '')} : {g.get('title', '')} {g.get('url', '')}",
                metadata={
                    "dataType": g.get("dataType", ""),
                    "type": g.get("type", ""),
                    "subject": g.get("subject", ""),
                    "title": g.get("title", ""),
                    "category": g.get("category", ""),
                    "summary": g.get("summary", ""),
                    "url": g.get("url", ""),
                    "guideid": g.get("guideid", ""),
                },
            )
        )

    logger.info(f"Nombre total de documents: {len(documents)}")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    logger.info(f"Nombre de chunks après découpage: {len(splits)}")

    # Créer des embeddings avec LangChain
    embedding_model = HuggingFaceEmbeddings(
        model_name=model_name,
    )
    vector_store = FAISS.from_documents(splits, embedding_model)

    elapsed_time = time.time() - start_time
    logger.info(f"Indexation terminée en {elapsed_time:.2f} secondes")

    return vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 2, "score_threshold": 0.3},
    )

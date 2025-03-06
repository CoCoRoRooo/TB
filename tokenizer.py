import json
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document


# Charger les guides depuis un fichier JSON
def load_guides(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        guides = json.load(f)
    return guides


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

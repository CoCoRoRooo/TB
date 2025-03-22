import json
import re
import ast
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


# Charger les guides depuis un fichier JSON
def load_guides(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        guides = json.load(f)
    return guides


def index_guides(guides, model_name="sentence-transformers/all-mpnet-base-v2"):
    documents = []

    for g in guides:
        documents.append(
            Document(
                page_content=f"{g['dataType']} - {g['type']} {g['subject']} : {g['title']} ({g['url']})",
                metadata={
                    "url": g["url"],
                    "type": g["type"],
                    "subject": g["subject"],
                    "title": g["title"],
                    "guideid": g["guideid"],
                },
            )
        )

    print(f"{len(documents)} documents créés.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)

    embedding_model = HuggingFaceEmbeddings(model_name=model_name)
    vector_store = FAISS.from_documents(splits, embedding_model)

    return vector_store.as_retriever(search_kwargs={"k": 2}), embedding_model

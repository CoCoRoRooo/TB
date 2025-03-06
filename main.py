import os
import bs4
import numpy as np
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

# Charger les variables d'environnement à partir du fichier .env
load_dotenv()

# 🔑 Configurer les API Keys (remplace "<your-api-key>" par ta clé si nécessaire)
OPENAI_KEY = os.getenv("OPENAI_KEY")

print("🔑 OPENAI_KEY :", OPENAI_KEY)

if not OPENAI_KEY:
    raise ValueError(
        "❌ OPENAI_KEY n'est pas défini ! Assure-toi de l'ajouter dans tes variables d'environnement."
    )

# 📌 1️⃣ CHARGEMENT DES DOCUMENTS
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

# 📌 2️⃣ FRACTIONNEMENT DU TEXTE
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# 📌 3️⃣ EMBEDDING & INDEXATION DANS CHROMADB
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=OpenAIEmbeddings(openai_api_key=OPENAI_KEY),  # Ajout explicite de la clé
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

# 📌 4️⃣ CONSTRUCTION DU PROMPT
prompt_template = """Réponds à la question uniquement avec le contexte suivant :
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(prompt_template)

# 📌 5️⃣ CONFIGURATION DU LLM (GPT-3.5)
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo", temperature=0, openai_api_key=OPENAI_KEY
)  # Ajout explicite de la clé

# 📌 6️⃣ CHAINAGE DU PROCESSUS (RAG CHAIN)
rag_chain = (
    {
        "context": retriever
        | (lambda docs: "\n\n".join(doc.page_content for doc in docs)),
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)

# 📌 7️⃣ TEST : POSER UNE QUESTION
question = "What is Task Decomposition?"
response = rag_chain.invoke(question)

print("📝 Réponse générée :")
print(response)

# import os
# import json
# import faiss
# import numpy as np
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings import OpenAIEmbeddings
# from langchain.chat_models import ChatOpenAI
# from langchain.prompts import ChatPromptTemplate
# from langchain.schema import Document
# from dotenv import load_dotenv

# load_dotenv()

# # 🔑 Configurer les API Keys
# OPENAI_KEY = os.getenv("OPENAI_KEY")

# if not OPENAI_KEY:
#     raise ValueError(
#         "❌ OPENAI_KEY n'est pas défini ! Assure-toi de l'ajouter dans tes variables d'environnement."
#     )

# # 📌 1️⃣ CHARGEMENT DES DOCUMENTS
# with open("data/guides.json", "r", encoding="utf-8") as f:
#     guides = json.load(f)

# # 📌 2️⃣ FRACTIONNEMENT DU TEXTE
# contents = [
#     f"{g['dataType']} - {g['type']} {g['subject']} : {g['title']} {(g['url'])}"
#     for g in guides
# ]

# # Fractionner le contenu
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# documents = [Document(page_content=content) for content in contents]
# splits = text_splitter.split_documents(documents)

# # 📌 3️⃣ EMBEDDING & INDEXATION AVEC FAISS
# embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_KEY)
# embedding_vectors = []

# # Convertir les documents en embeddings et les stocker dans embedding_vectors
# for content in contents:
#     embedding = embeddings.embed_documents([content])[0]
#     embedding_vectors.append(embedding)

# # Convertir la liste d'embeddings en une matrice NumPy
# embedding_matrix = np.array(embedding_vectors).astype("float32")

# # Créer un index FAISS et l'ajouter aux embeddings
# dimension = embedding_matrix.shape[1]  # Dimension des embeddings
# faiss_index = faiss.IndexFlatL2(dimension)
# faiss_index.add(embedding_matrix)  # Ajouter les embeddings à l'index FAISS

# # 📌 4️⃣ CONSTRUCTION DU PROMPT
# prompt_template = """Réponds à la question en français uniquement avec le contexte suivant :
# {context}

# Question: {question}
# """
# prompt = ChatPromptTemplate.from_template(prompt_template)

# # 📌 5️⃣ CONFIGURATION DU LLM (GPT-3.5)
# llm = ChatOpenAI(
#     model_name="gpt-3.5-turbo", temperature=0, openai_api_key=OPENAI_KEY
# )

# # 📌 6️⃣ CHAINAGE DU PROCESSUS (RAG CHAIN)
# def retrieve_context(question):
#     # Convertir la question en embedding
#     question_embedding = embeddings.embed_documents([question])[0]
#     question_embedding = np.array([question_embedding]).astype("float32")

#     # Rechercher dans FAISS les documents les plus proches de la question
#     distances, indices = faiss_index.search(question_embedding, k=1)  # Recherche des 1 voisins les plus proches
#     context = "\n\n".join([splits[idx].page_content for idx in indices[0]])  # Récupérer les contenus associés
#     return context

# # 📌 7️⃣ TEST : POSER UNE QUESTION
# question = "Comment réparer un clavier ?"
# context = retrieve_context(question)

# # Construire le prompt avec le contexte récupéré
# final_prompt = prompt.format(context=context, question=question)

# # Générer la réponse
# response = llm(final_prompt)

# print("📝 Réponse générée :")
# print(response)

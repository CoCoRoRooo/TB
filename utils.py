from json import dumps, loads
from langchain.schema import Document


def format_documents(docs):
    """Formate les documents pour l'affichage"""
    from api_client import get_guide_steps

    formatted_docs = []
    for doc in docs:
        guide_id = doc.metadata.get("guideid")
        if guide_id:
            guide_steps = get_guide_steps(guide_id)
            guide_infos = ""
            for guide in guide_steps:
                step_text = "\n".join(guide["text"])
                guide_infos += "\n" + f"Step {guide['stepno']}:\n" + step_text

            if guide_infos not in doc.page_content:
                doc.page_content += guide_infos

        metadata_text = "\n".join(
            f"{key}: {value}" for key, value in doc.metadata.items()
        )
        formatted_doc = f"""---\n📄 Contenu :\n{doc.page_content}\n\n🔖 Métadonnées :\n{metadata_text}\n"""
        formatted_docs.append(formatted_doc)

    return "\n\n".join(formatted_docs)


def get_unique_union(documents: list[list[Document]]) -> list[Document]:
    """Renvoie une liste de documents uniques à partir d'une liste de listes de documents."""
    # Aplatir la liste
    flattened_docs = [
        dumps(doc.__dict__, sort_keys=True) for sublist in documents for doc in sublist
    ]

    # Supprimer les doublons
    unique_docs = list(set(flattened_docs))

    # Reconvertir en objets Document
    return [Document(**loads(doc)) for doc in unique_docs]

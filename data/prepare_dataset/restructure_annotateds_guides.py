import json
import re


def split_annotations(text):
    """
    Sépare le texte en trois parties :
    - "general_problem" avant "Symptoms"
    - "symptoms" entre "Symptoms" et "Solution"
    - "general_solution" après "Solution"

    La comparaison est insensible à la casse.
    """
    # Transformer en minuscule pour la recherche des délimiteurs
    text_lower = text.lower()

    # Trouver les positions des délimiteurs
    symptoms_index = text_lower.find("symptoms")
    solution_index = text_lower.find("solution")

    # Initialiser les sections
    general_problem = (
        text[:symptoms_index].strip() if symptoms_index != -1 else text.strip()
    )
    symptoms = (
        text[symptoms_index:solution_index].strip()
        if symptoms_index != -1 and solution_index != -1
        else ""
    )
    general_solution = text[solution_index:].strip() if solution_index != -1 else ""

    return general_problem, symptoms, general_solution


def restructure_guides(guides):
    for guide in guides:
        if "annotations" in guide:
            annotations = guide["annotations"].strip()

            # Séparer les sections
            general_problem, symptoms, general_solution = split_annotations(annotations)

            # Ajouter les nouvelles clés
            guide["general_problem"] = general_problem
            guide["symptoms"] = symptoms
            guide["general_solution"] = general_solution

            # Supprimer l'ancienne clé "annotations"
            del guide["annotations"]

    return guides


# Charger le fichier JSON d'entrée
with open("data/prepare_dataset/guides_annotated.json", "r", encoding="utf-8") as f:
    guides = json.load(f)

# Transformation des données
new_guides = restructure_guides(guides)

# Sauvegarde du fichier JSON modifié
with open("data/prepare_dataset/guides_dataset.json", "w", encoding="utf-8") as f:
    json.dump(new_guides, f, indent=4, ensure_ascii=False)

print("Guide restructuré avec succès !")

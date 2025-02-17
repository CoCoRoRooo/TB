import json
import ollama
import time

# Charger le fichier guides.json
with open("data/guides.json", "r", encoding="utf-8") as f:
    guides = json.load(f)


# Fonction pour générer les annotations avec Mistral
def generate_annotations(specific_solution):
    prompt = f"""
    Here's a specific solution for a technical problem: "{specific_solution}".

    Generates a **English** response in **strict JSON form** with **these 3 keys**:
    - `"general_problem"` (string): A short sentence describing the problem without mentioning the guide.
    - `"symptoms"` (list of strings): A list of symptoms.
    - `"general_solution"` (string): A short sentence describing a potential solution without mentioning the guide.
    """

    response = ollama.chat(
        model="mistral", messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"]


# Annoter les guides
enriched_guides = []
for guide in guides:
    specific_solution = f"{guide['dataType']} - {guide['subject']} {guide['type']} {guide['category']} : {guide['title']} ({guide['url']})"
    if specific_solution:
        try:
            annotations = generate_annotations(specific_solution)
            guide["annotations"] = annotations
            print(guide["annotations"] + "\n" + str(guide["guideid"]))
        except Exception as e:
            print(f"Erreur avec {specific_solution}: {e}")
            guide["annotations"] = "Erreur de génération"

    enriched_guides.append(guide)
    # time.sleep(1)  # Éviter de surcharger l'API

# Sauvegarder le fichier enrichi
with open("data/prepare_dataset/guides_annotated.json", "w", encoding="utf-8") as f:
    json.dump(enriched_guides, f, indent=4, ensure_ascii=False)

print("Annotations générées et sauvegardées avec succès !")

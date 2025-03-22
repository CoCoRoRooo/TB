import json
import ollama
import time

# Charger le fichier guides.json
with open("data/guides.json", "r", encoding="utf-8") as f:
    guides = json.load(f)


# Fonction pour générer les annotations avec Mistral
def generate_annotations(specific_solution):
    prompt = f"""
    Voici une solution spécifique pour un problème technique : "{specific_solution}".

    Génère et retourne uniquement une réponse en **anglais** sous **forme stricte de JSON** avec **ces 3 clés** :
    - `"user_problem"` (list de strings) : Une liste de potentiels questions évoqués par un utilisateur pour arriver à cette solution.
    - `"user_symptoms"` (list de strings) : Une liste de potentiels symptômes évoqués par un utilisateur pour arriver à cette solution.
    """

    response = ollama.chat(
        model="llama3", messages=[{"role": "user", "content": prompt}]
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
        except Exception as e:
            print(f"Erreur avec {specific_solution}: {e}")
            guide["annotations"] = "Erreur de génération"

    enriched_guides.append(guide)
    print(
        str(enriched_guides[-1]["guideid"]) + " " + enriched_guides[-1]["annotations"]
    )
    # time.sleep(1)  # Éviter de surcharger l'API

# Sauvegarder le fichier enrichi
with open("data/OLD/guides_annotated.json", "w", encoding="utf-8") as f:
    json.dump(enriched_guides, f, indent=4, ensure_ascii=False)

print("Annotations générées et sauvegardées avec succès !")

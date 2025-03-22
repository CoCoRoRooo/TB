import time
import requests
import json
import os

# Définition des constantes
BASE_URL = "https://www.ifixit.com/api/2.0"
GUIDES_FILE = "./data/guides.json"

# Création du dossier de sortie s'il n'existe pas
os.makedirs(os.path.dirname(GUIDES_FILE), exist_ok=True)


def fetch_all_guides():
    """Récupère la liste de tous les guides avec leurs métadonnées."""
    all_guides = []
    offset = 0
    limit = 20
    has_more = True

    while has_more:
        url = f"{BASE_URL}/guides?offset={offset}&limit={limit}"
        response = requests.get(url)

        # Vérifiez la réponse
        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            try:
                guides = response.json()
                all_guides.extend(guides)

                if len(guides) < limit:
                    has_more = False
                else:
                    offset += limit
            except ValueError:
                print("Erreur de décodage JSON.")
                break
        elif response.status_code == 429:
            print("Limite de requêtes atteinte. Attente de 30 secondes...")
            time.sleep(30)  # Attente avant de réessayer
        else:
            print(f"Erreur {response.status_code} lors de la récupération des guides.")
            break

    return all_guides


def fetch_guide_details(guide_id):
    """Récupère le contenu détaillé d'un guide à partir de son guideid."""
    url = f"{BASE_URL}/guides/{guide_id}"
    response = requests.get(url)

    if response.status_code == 200:
        try:
            return response.json()
        except ValueError:
            print(f"Erreur de décodage JSON pour le guide {guide_id}")
    elif response.status_code == 429:
        print("Limite de requêtes atteinte. Attente de 30 secondes...")
        time.sleep(30)  # Attente avant de réessayer
        return fetch_guide_details(guide_id)  # Nouvelle tentative
    else:
        print(
            f"Erreur {response.status_code} lors de la récupération du guide {guide_id}"
        )

    return None


def fetch_all_guides_with_details():
    """Récupère tous les guides avec leurs métadonnées et contenu détaillé."""
    guides_metadata = fetch_all_guides()
    all_guides_data = []

    for guide in guides_metadata:
        guide_id = guide.get("guideid")
        if guide_id:
            print(f"Récupération du contenu du guide {guide_id}...")
            details = fetch_guide_details(guide_id)
            if details:
                all_guides_data.append(details)

    return all_guides_data


# Récupération de tous les guides avec leurs détails
guides_with_details = fetch_all_guides_with_details()

# Enregistrement des guides enrichis dans un fichier JSON
if guides_with_details:
    with open(GUIDES_FILE, "w", encoding="utf-8") as f:
        json.dump(guides_with_details, f, indent=2, ensure_ascii=False)
    print(f"Données enregistrées dans {GUIDES_FILE}")
else:
    print("Aucun guide trouvé ou problème avec l'API.")

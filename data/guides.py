import time
import requests
import json


def fetch_all_guides():
    all_guides = []
    offset = 0
    limit = 20
    has_more = True

    while has_more:
        url = f"https://www.ifixit.com/api/2.0/guides?offset={offset}&limit={limit}"
        response = requests.get(url)

        # Vérifiez la réponse
        print(f"Status Code: {response.status_code}")
        print(
            f"Response Content: {response.content[:200]}..."
        )  # Affiche les premiers 200 caractères de la réponse

        if response.status_code == 200:
            try:
                guides = response.json()
                all_guides.extend(guides)

                if len(guides) < limit:
                    has_more = False
                else:
                    offset += limit
            except ValueError:
                print("Erreur de décodage JSON")
                break
        elif response.status_code == 429:
            # Si la limite est atteinte, attendons 30 secondes avant de continuer
            print("Limite de requêtes atteinte. Attente de 30 secondes...")
            time.sleep(30)  # Attendre 30 secondes avant de réessayer
        else:
            print(
                f"Erreur lors de la récupération des données. Code statut: {response.status_code}"
            )
            break

    return all_guides


# Récupérer tous les guides
guides = fetch_all_guides()

# Enregistrement des guides dans un fichier JSON
if guides:
    with open("./data/guides.json", "w") as f:
        json.dump(guides, f, indent=2)
    print("Données enregistrées dans guides.json")
else:
    print("Aucun guide trouvé ou problème avec l'API.")

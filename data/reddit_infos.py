import requests
import json

# URL de base du subreddit 'techsupport' pour récupérer les posts en JSON
url_base = "https://www.reddit.com/r/techsupport/new/.json"

# En-têtes pour simuler un navigateur, car Reddit bloque les requêtes sans user-agent
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

# Liste pour stocker les problèmes et solutions extraits
scraped_data = []

# Paramètres de pagination
limit = 20  # Limite des résultats par page
after = None  # Initialement, aucune pagination

# Boucle pour récupérer toutes les pages
while True:
    # Ajouter l'offset 'after' à l'URL si nécessaire
    url = f"{url_base}?limit={limit}"
    if after:
        url += f"&after={after}"

    # Effectuer la requête HTTP pour récupérer les posts en JSON
    response = requests.get(url, headers=headers)

    # Vérifier si la requête a réussi
    if response.status_code == 200:
        data = response.json()

        # Boucler sur les posts pour extraire les problèmes et solutions
        for post in data["data"]["children"]:
            title = post["data"]["title"]  # Problème (titre du post)
            content = post["data"][
                "selftext"
            ]  # Contenu du post (description du problème)
            url = post["data"]["url"]  # URL du post comme solution ou source

            # Ajouter les données dans la liste
            scraped_data.append(
                {
                    "problème": {"titre": title, "description": content},
                    "solution": url,
                    "source": "Reddit - r/techsupport",
                }
            )

        # Vérifier s'il y a une page suivante
        after = data["data"].get("after")

        # Si 'after' est None, cela signifie qu'il n'y a plus de pages
        if not after:
            break
    else:
        print(f"Erreur lors de la récupération des données: {response.status_code}")
        break

# Sauvegarder les données extraites dans un fichier JSON
with open("./data/reddit_scraped_data.json", "w", encoding="utf-8") as json_file:
    json.dump(scraped_data, json_file, ensure_ascii=False, indent=4)

print("Les données ont été sauvegardées dans 'reddit_scraped_data.json'")

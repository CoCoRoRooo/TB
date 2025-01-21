from googleapiclient.discovery import build
from dotenv import load_dotenv
import os

# Configuration de l'API YouTube
load_dotenv()

YTB_DATA_API_V3_KEY = os.getenv("YTB_DATA_API_V3_KEY")
youtube = build("youtube", "v3", developerKey=YTB_DATA_API_V3_KEY)


def search_youtube_videos(keyword, max_results=3):
    """
    Recherche des vidéos YouTube basées sur un mot-clé.

    :param keyword: Mot-clé pour la recherche.
    :param max_results: Nombre maximum de résultats à récupérer.
    :return: Liste des vidéos trouvées.
    """
    try:
        print(f"Recherche de vidéos YouTube pour : {keyword}")
        # Appel de l'API YouTube pour effectuer la recherche
        search_response = (
            youtube.search()
            .list(
                q=keyword,
                part="snippet",
                type="video",  # Recherche uniquement des vidéos
                maxResults=max_results,
            )
            .execute()
        )

        # Extraire les données pertinentes des résultats
        videos = []
        for item in search_response.get("items", []):
            # Construction de l'objet vidéo basé sur la réponse de l'API
            video = {
                "video_id": item["id"]["videoId"],
                "title": item["snippet"]["title"],
                "description": item["snippet"]["description"],
                "channel_title": item["snippet"]["channelTitle"],
                "published_at": item["snippet"]["publishedAt"],
                "thumbnail_url": item["snippet"]["thumbnails"]["default"][
                    "url"
                ],  # Ajout de la miniature
                "channel_id": item["snippet"]["channelId"],  # Identifiant du canal
                "url": f"https://www.youtube.com/watch?v={item['id']['videoId']}",  # URL de la vidéo
            }
            videos.append(video)

        return videos

    except Exception as e:
        print(f"Erreur lors de la recherche : {e}")
        return []

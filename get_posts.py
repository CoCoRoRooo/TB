import praw
import json

# 🔑 Connexion à l'API Reddit
reddit = praw.Reddit(
    client_id="6NAkftfhMW0UzBTsOk_WUw",
    client_secret="DGoO-wC2rWF9kuHRk8EcvQ5IIvoq2Q",
    user_agent="script:reddit.techsupport.scraper:v1.0 (par /u/Ambitious_Shopping45)",
)

# 📂 Fichier JSON pour stocker les posts avec leurs commentaires
FICHIER_JSON = "techsupport_posts.json"

# 📡 Accéder au subreddit
subreddit = reddit.subreddit("techsupport")

print("🚀 Récupération des posts et de leurs commentaires...")

# Récupérer un maximum de posts
posts_data = []

for post in subreddit.top(limit=None):  # Pas de limite de posts
    post.comments.replace_more(limit=0)  # Supprime les "More Comments"

    # Récupérer jusqu'à 50 commentaires par post
    comments = [comment.body for comment in post.comments.list()[:20]]

    post_info = {
        "id": post.id,
        "titre": post.title,
        "contenu": post.selftext,
        "url": post.url,
        "date": post.created_utc,
        "score": post.score,
        "nombre_commentaires": post.num_comments,
        "comments": comments,  # 🔥 Liste des 20 premiers commentaires max
    }

    posts_data.append(post_info)

# 💾 Sauvegarde en une seule fois (plus rapide)
with open(FICHIER_JSON, "w", encoding="utf-8") as f:
    json.dump(posts_data, f, indent=4, ensure_ascii=False)

print(
    f"✅ {len(posts_data)} posts enregistrés avec un max de 20 commentaires chacun dans {FICHIER_JSON} !"
)

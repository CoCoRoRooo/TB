from openai import OpenAI
from dotenv import load_dotenv
import os
import ollama

# Charger les variables d'environnement à partir du fichier .env
load_dotenv()

# Initialisation du client avec clé API OpenAI
OPENAI_KEY = os.getenv("OPENAI_KEY")
client = OpenAI(api_key=OPENAI_KEY)


def generate_response(user_query, relevant_guides):
    context = "\n\n".join(relevant_guides)
    prompt = f"L'utilisateur pose la question : {user_query}\n\nVoici des guides pertinents :\n{context}\n\nRéponds de manière concise et utile en t'appuyant sur les guides s'il y en a."

    print(prompt)

    # response = client.chat.completions.create(
    #     model="gpt-4",  # Utiliser un modèle plus léger si besoin (gpt-3.5-turbo)
    #     messages=[
    #         {"role": "system", "content": "Tu es un assistant technique."},
    #         {"role": "user", "content": prompt},
    #     ],
    #     temperature=0.5,
    # )

    response = ollama.chat(
        model="mistral", messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]

    # return response.choices[0].message.content

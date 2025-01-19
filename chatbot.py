from openai import OpenAI
from dotenv import load_dotenv
from utils.conversation_manager import ConversationManager
from interface.user_interface import ask_user_question
import os

# Charger les variables d'environnement à partir du fichier .env
load_dotenv()

# Initialisation du client avec votre clé API OpenAI
OPENAI_KEY = os.getenv("OPENAI_KEY")
client = OpenAI(api_key=OPENAI_KEY)


def start_conversation():
    """
    Lance la conversation avec une question initiale et ajoute un contexte à l'assistant.
    """
    # Message pour définir le rôle de l'assistant
    assistant_context = (
        "Vous êtes un assistant expert en appareils électroniques et en réparation. "
        "Votre rôle est de diagnostiquer les problèmes rencontrés par les utilisateurs "
        "et de fournir des solutions à partir d'informations disponibles sur internet, "
        "y compris des forums, blogs, vidéos, guides, et autres ressources. "
        "Vous pouvez également recommander des guides de réparation spécifiques "
        "pour aider les utilisateurs à résoudre leurs problèmes. "
    )

    # Ajout du message d'introduction au contexte de la conversation
    conversation_manager.add_message("assistant", assistant_context)

    # Question initiale
    initial_question = "Quel est le problème principal que vous rencontrez ?"
    conversation_manager.add_message("assistant", initial_question)

    return initial_question


# Instances globales
conversation_manager = ConversationManager()

# Démarrage de la conversation avec la première question
start_conversation()


# Fonction pour interagir avec l'API OpenAI
def chat_gpt(prompt):
    """
    Interagit avec l'API OpenAI pour générer une réponse basée sur un prompt donné.

    Args:
        prompt (str): Le texte ou la question pour GPT.

    Returns:
        str: La réponse générée par GPT.
    """
    # Récupérer l'historique des messages existants
    messages = conversation_manager.get_history()

    # Ajoutez le message utilisateur à l'historique
    conversation_manager.add_message("user", prompt)

    # Incluez le nouveau message utilisateur
    messages.append({"role": "user", "content": prompt})

    try:
        # Appel à l'API OpenAI
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
        )

        # Ajoutez la réponse de l'assistant à l'historique
        assistant_response = response.choices[0].message.content.strip()
        conversation_manager.add_message("assistant", assistant_response)

        return assistant_response
    except Exception as e:
        return f"Une erreur est survenue avec l'API OpenAI : {e}"


def get_recommended_guides():
    guides_text_data = ""
    for message in reversed(conversation_manager.get_history()):
        if message["role"] == "assistant":
            guides_text_data = message["content"]
            break  # Arrête la boucle dès qu'on trouve le dernier message de l'assistant

    print(guides_text_data)

    # Demander des recommandations de guides basées sur l'historique
    return ask_user_question(guides_text_data)

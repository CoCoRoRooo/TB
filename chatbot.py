from openai import OpenAI
from interface.user_interface import ask_user_question

# Initialisation du client avec votre clé API OpenAI
client = OpenAI(
    api_key="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"
)


# Fonction pour interagir avec l'API OpenAI
def chat_gpt(prompt, context=None):
    """
    Interagit avec l'API OpenAI pour générer une réponse basée sur un prompt donné.

    Args:
        prompt (str): Le texte ou la question pour GPT.
        context (str, optional): Un contexte supplémentaire pour personnaliser la réponse.

    Returns:
        str: La réponse générée par GPT.
    """
    messages = []
    if context:
        messages.append({"role": "system", "content": context})
    messages.append({"role": "user", "content": prompt})

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Une erreur est survenue avec l'API OpenAI : {e}"


# Exemple d'utilisation
if __name__ == "__main__":
    question = "Réparer un écran d'Imac cassé"
    # Contexte pour guider GPT
    context = "Tu es un assistant spécialisé dans les réparations électroniques. Aide l'utilisateur à identifier les ressources utiles pour effectuer les réparations répond sous forme de liste à points."
    # Obtenir une réponse initiale de GPT à la question
    response = chat_gpt(question, context)
    print("Réponse initiale de GPT :")
    print(response)
    ask_user_question(question)

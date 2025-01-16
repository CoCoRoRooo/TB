class ConversationManager:
    def __init__(self):
        self.history = []

    def add_message(self, role, content):
        """
        Ajoute un message au contexte historique.

        Args:
            role (str): "user" ou "assistant".
            content (str): Le contenu du message.
        """
        self.history.append({"role": role, "content": content})

    def get_history(self):
        """
        Récupère l'historique complet des messages.

        Returns:
            list: L'historique des messages.
        """
        return self.history

    def clear_history(self):
        """Réinitialise l'historique des conversations."""
        self.history = []

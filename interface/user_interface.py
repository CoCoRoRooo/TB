from prediction.recommendation_system import recommend_guides
from deep_translator import GoogleTranslator


def ask_user_question(user_question):
    translated_question = GoogleTranslator(source="auto", target="en").translate(
        user_question
    )
    print(f"Question de l'utilisateur : {translated_question}")
    recommended_guides = recommend_guides(translated_question)

    for recommended_guide in recommended_guides:
        print(f"Guide recommandé : {recommended_guide['title']}")
        print(f"Résumé : {recommended_guide['summary']}")
        print(f"URL : {recommended_guide['url']}")

    return {"recommended_guides": recommended_guides}

from prediction.recommendation_system import recommend_guides
from deep_translator import GoogleTranslator
from prediction.ytb_videos import search_youtube_videos


def get_resources(chatbot_infos):
    translated_chatbot_infos = GoogleTranslator(source="auto", target="en").translate(
        chatbot_infos
    )

    print(f"Translated chatbot infos: {translated_chatbot_infos}")

    recommended_guides = recommend_guides(translated_chatbot_infos)

    ytb_videos = []

    for recommended_guide in recommended_guides:
        ytb_videos.append(
            search_youtube_videos(
                f"{recommended_guide["title"]} {recommended_guide['category']}"
            )
        )

    return {"recommended_guides": recommended_guides, "ytb_videos": ytb_videos}

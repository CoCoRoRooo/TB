from prediction.predict import predict_guides
from models.load_data import load_guides


def recommend_guides(chatbot_infos, guides_filepath="data/guides.json"):
    guides = load_guides(guides_filepath)
    return predict_guides(chatbot_infos, guides, top_n=3)

from prediction.predict import predict_guides
from models.load_data import load_guides


def recommend_guides(user_question, guides_filepath="guides.json"):
    guides = load_guides(guides_filepath)
    return predict_guides(user_question, guides, top_n=3)

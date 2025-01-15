import json


def load_guides(filepath):
    with open(filepath, "r") as file:
        guides = json.load(file)
    return guides

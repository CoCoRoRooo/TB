import requests
import logging

logger = logging.getLogger(__name__)


def get_guide_steps(guideid):
    """Récupère les étapes d'un guide depuis l'API iFixit"""
    url = f"https://www.ifixit.com/api/2.0/guides/{guideid}"

    try:
        response = requests.get(url)
        if response.status_code != 200:
            logger.warning(
                f"Échec de récupération du guide {guideid}, code: {response.status_code}"
            )
            return {
                "error": f"Échec de récupération du guide {guideid}, code: {response.status_code}"
            }

        data = response.json()
        steps = []
        cpt_steps = 0

        for step in data.get("steps", []):
            cpt_steps += 1
            step_texts = [
                line["text_rendered"]
                for line in step.get("lines", [])
                if "text_rendered" in line
            ]
            steps.append({"stepno": cpt_steps, "text": step_texts})

        logger.info(f"Guide {guideid} récupéré avec succès: {len(steps)} étapes")
        return steps

    except Exception as e:
        logger.error(f"Erreur lors de la récupération du guide {guideid}: {e}")
        return []

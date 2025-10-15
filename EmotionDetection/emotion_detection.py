import requests
import json

URL = "https://sn-watson-emotion.labs.skills.network/v1/watson.runtime.nlp.v1/NlpService/EmotionPredict"
HEADERS = {"grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"}

_EMPTY_RESULT = {
    "anger": None,
    "disgust": None,
    "fear": None,
    "joy": None,
    "sadness": None,
    "dominant_emotion": None,
}

def emotion_detector(text_to_analyze: str):
    """
    Envía el texto a la API EmotionPredict y retorna un dict con:
    anger, disgust, fear, joy, sadness y dominant_emotion.
    Si la API responde 400 (entrada vacía), retorna todas las claves en None.
    """
    payload = {"raw_document": {"text": text_to_analyze if text_to_analyze is not None else ""}}
    resp = requests.post(URL, headers=HEADERS, json=payload, timeout=30)

    # Requisito de la Tarea 7: usar status_code y, si es 400, devolver None en todo
    if resp.status_code == 400:
        return _EMPTY_RESULT

    data = json.loads(resp.text)
    emotions = data["emotionPredictions"][0]["emotion"]

    anger   = emotions.get("anger", 0.0)
    disgust = emotions.get("disgust", 0.0)
    fear    = emotions.get("fear", 0.0)
    joy     = emotions.get("joy", 0.0)
    sadness = emotions.get("sadness", 0.0)

    scores = {"anger": anger, "disgust": disgust, "fear": fear, "joy": joy, "sadness": sadness}
    dominant = max(scores, key=scores.get)

    return {
        "anger": anger,
        "disgust": disgust,
        "fear": fear,
        "joy": joy,
        "sadness": sadness,
        "dominant_emotion": dominant,
    }

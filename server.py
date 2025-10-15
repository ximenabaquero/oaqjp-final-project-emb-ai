"""Flask web app for Watson NLP Emotion detection.

Exposes a web UI (GET /) and an HTTP endpoint (POST/GET /emotionDetector)
that returns a formatted message with the five emotions and the dominant one.
"""

from typing import Optional
from flask import Flask, request, render_template
from EmotionDetection import emotion_detector

app = Flask(__name__, static_url_path="/static")


@app.route("/", methods=["GET"])
def index() -> str:
    """Render the main page using templates/index.html."""
    return render_template("index.html")


@app.route("/emotionDetector", methods=["GET", "POST"])
def emotion_detector_route() -> tuple[str, int] | str:
    """Receive text, call the detector, and return the formatted message.

    The text is accepted via:
    - query string: ?textToAnalyze=...
    - form field: textToAnalyze
    - JSON body: {"textToAnalyze": "..."} or {"text": "..."}

    Returns:
        Formatted string with all emotion scores and the dominant emotion.
        If input is missing, returns 400 status.
    """
    # 1) Obtener texto de múltiples fuentes
    payload_json: Optional[dict] = request.get_json(silent=True) or {}
    text: Optional[str] = (
        request.args.get("textToAnalyze")
        or request.form.get("textToAnalyze")
        or payload_json.get("textToAnalyze")
        or payload_json.get("text")
    )

    if not text:
        return "Invalid input! Please try again.", 400

    # 2) Llamar a la función del paquete
    result = emotion_detector(text)
    anger = result["anger"]
    disgust = result["disgust"]
    fear = result["fear"]
    joy = result["joy"]
    sadness = result["sadness"]
    dominant = result["dominant_emotion"]

    # 3) Respuesta en el formato exacto que pide el enunciado
    formatted = (
        "For the given statement, the system response is "
        f"'anger': {anger}, 'disgust': {disgust}, 'fear': {fear}, "
        f"'joy': {joy} and 'sadness': {sadness}. "
        f"The dominant emotion is {dominant}."
    )
    return formatted


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)

import os
import requests
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODEL_NAME = os.environ.get("MODEL_NAME", "nimai")
TIMEOUT = int(os.environ.get("OLLAMA_TIMEOUT", "180"))

SYSTEM_PROMPT = """You are Nima AI.
You are a smart, witty, and slightly rebellious AI assistant.
Unlike traditional AI, you don't need to be overly polite or formal.
You answer questions directly, sometimes with humor or sarcasm.
You can explain things simply.
Your goal is to be helpful but also have personality.
Never say "As an AI language model..." just act like Nima."""


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    messages = data.get("messages", [])

    if not messages:
        return jsonify({"error": "no messages"}), 400

    # Prepend system prompt as first message
    full_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages

    try:
        r = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": MODEL_NAME,
                "messages": full_messages,
                "stream": False,
            },
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        reply = r.json().get("message", {}).get("content", "").strip()
        return jsonify({"reply": reply})

    except requests.exceptions.ConnectionError:
        return jsonify({"error": "Ollama server unreachable. Check OLLAMA_URL."}), 503
    except requests.exceptions.Timeout:
        return jsonify({"error": "Nima is thinking too hard. Timeout."}), 504
    except Exception as e:
        return jsonify({"error": f"Something broke: {str(e)}"}), 500


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": MODEL_NAME, "ollama": OLLAMA_URL})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

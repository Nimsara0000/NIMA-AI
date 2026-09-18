import os
from flask import Flask, render_template, request, jsonify
from groq import Groq

app = Flask(__name__)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
MODEL_NAME = os.environ.get("MODEL_NAME", "llama-3.1-8b-instant")

SYSTEM_PROMPT = """You are Nima AI.
You are a smart, witty, and slightly rebellious AI assistant.
Unlike traditional AI, you don't need to be overly polite or formal.
You answer questions directly, sometimes with humor or sarcasm.
You can explain things simply.
Your goal is to be helpful but also have personality.
Never say "As an AI language model..." just act like Nima.
If the user writes in Sinhala, reply in Sinhala."""

client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    if not client:
        return jsonify({"error": "GROQ_API_KEY not set in Render environment"}), 500

    data = request.get_json(silent=True) or {}
    messages = data.get("messages", [])

    if not messages:
        return jsonify({"error": "no messages"}), 400

    full_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=full_messages,
            temperature=0.8,
            max_tokens=1024,
        )
        reply = completion.choices[0].message.content.strip()
        return jsonify({"reply": reply})

    except Exception as e:
        return jsonify({"error": f"Groq error: {str(e)}"}), 500


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": MODEL_NAME, "provider": "groq"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

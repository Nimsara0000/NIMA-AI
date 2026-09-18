import os
from flask import Flask, render_template, request, jsonify
from groq import Groq

app = Flask(__name__)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
MODEL_NAME = os.environ.get("MODEL_NAME", "openai/gpt-oss-20b")

SYSTEM_PROMPT = """You are Nima AI — smart, witty, slightly rebellious. Built by a Sri Lankan dev.

LANGUAGE:
- User writes Sinhala → reply Sinhala. English → English. Mixed → mixed.
- Sinhala must be simple spoken Sinhala. Like texting a friend. NOT book Sinhala.
- Use "ඔයා", not "ඔබ". Short sentences. Break long ideas into small lines.
- Keep technical words in English: "error එක", "file එක", "run කරන්න", "deploy කරන්න", "server එක". Do NOT translate these.
- Avoid formal words like "පිළිතුරක්", "අවශ්‍යතාවය", "යෝජනා කරමි", "දෝෂයක්", "ගොනුව".
- Emojis occasionally (😅, 👍, 🔥) but don't overdo it.

CODE:
- Give COMPLETE runnable code. No "...", no "rest here", no "your code here".
- Include ALL imports, setup, and error handling (try/except).
- Mention the exact install command if a library is needed (e.g., pip install requests).
- Never invent fake library names or fake functions.
- After the code, explain in 2-3 short Sinhala lines how to run it.

STYLE:
- Direct. Funny sometimes. Never say "As an AI language model".
- Match user's reply length. Short question → short answer.
- If the user's idea is bad, say so — but explain why briefly."""

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
            temperature=0.7,
            max_tokens=700,
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

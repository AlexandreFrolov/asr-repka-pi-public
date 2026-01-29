import subprocess
from flask import Flask, request, send_file
import tempfile
import os

PIPER_BIN = "piper"
MODEL_PATH = "/root/piper-voices/ru/ru_RU-irina-medium.onnx"

app = Flask(__name__)

@app.route("/tts_wav", methods=["POST"])
def tts_wav():
    text = request.json.get("text", "").strip()
    if not text:
        return "Text is empty", 400

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        wav_path = f.name

    cmd = [
        PIPER_BIN,
        "--model", MODEL_PATH,
        "--voice", "irina",
        "--text", text,
        "--output_file", wav_path
    ]

    subprocess.run(cmd, check=True)

    return send_file(wav_path, mimetype="audio/wav", as_attachment=False)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

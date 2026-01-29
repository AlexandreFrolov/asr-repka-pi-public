import subprocess
from flask import Flask, request, Response, stream_with_context
import tempfile
import os

PIPER_BIN = "piper"
MODEL_PATH = "/root/piper-voices/ru/ru_RU-irina-medium.onnx"

app = Flask(__name__)

@app.route("/tts_stream", methods=["POST"])
def tts_stream():
    text = request.json.get("text", "").strip()
    if not text:
        return "Text is empty", 400

    # Разбиваем текст на короткие фрагменты (по точкам)
    fragments = [s.strip() for s in text.replace("!", ".").replace("?", ".").split(".") if s.strip()]

    def generate():
        for frag in fragments:
            # создаём временный WAV
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                wav_path = f.name

            cmd = [
                PIPER_BIN,
                "--model", MODEL_PATH,
                "--voice", "irina",
                "--text", frag,
                "--output_file", wav_path
            ]
            subprocess.run(cmd, check=True)

            # читаем WAV и отдаём клиенту
            with open(wav_path, "rb") as f:
                while True:
                    chunk = f.read(4096)
                    if not chunk:
                        break
                    yield chunk

            os.remove(wav_path)

    return Response(stream_with_context(generate()), mimetype="application/octet-stream")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

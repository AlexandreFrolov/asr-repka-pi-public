import requests
import sounddevice as sd
import numpy as np
import subprocess

SERVER_URL = "http://127.0.0.1:5000/tts_wav"
TEXT = "Привет! Это репка-пи проверка звука."

# ---- Получаем WAV с сервера ----
r = requests.post(SERVER_URL, json={"text": TEXT}, stream=True)
r.raise_for_status()

# ---- Воспроизведение через ffmpeg в stdout -> sounddevice ----
ffmpeg = subprocess.Popen(
    ["ffmpeg", "-i", "pipe:0", "-f", "s16le", "-ar", "22050", "-ac", "1", "pipe:1"],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE
)

for chunk in r.iter_content(4096):
    if chunk:
        ffmpeg.stdin.write(chunk)
ffmpeg.stdin.close()

# ---- Читаем PCM из ffmpeg ----
stream = sd.OutputStream(samplerate=22050, channels=1, dtype='int16')
stream.start()

while True:
    pcm_chunk = ffmpeg.stdout.read(4096)
    if not pcm_chunk:
        break
    audio = np.frombuffer(pcm_chunk, dtype=np.int16)
    stream.write(audio)

stream.stop()
stream.close()

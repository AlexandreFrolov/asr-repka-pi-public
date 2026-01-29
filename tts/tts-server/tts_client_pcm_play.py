import requests
import sounddevice as sd
import numpy as np
import subprocess

SERVER_URL = "http://127.0.0.1:5000/tts_stream"
TEXT = "Привет! Это рабочий потоковый TTS на Репка-Пи. Речь начинается почти сразу."

SAMPLE_RATE = 22050
CHANNELS = 1
DTYPE = 'int16'

# OutputStream для мгновенного воспроизведения
stream = sd.OutputStream(
    samplerate=SAMPLE_RATE,
    channels=CHANNELS,
    dtype=DTYPE,
    blocksize=512  # маленький блок → низкая задержка
)
stream.start()

# Запрос на сервер
with requests.post(SERVER_URL, json={"text": TEXT}, stream=True, timeout=60) as r:
    r.raise_for_status()

    # Запускаем ffmpeg для конвертации WAV → PCM
    ffmpeg = subprocess.Popen(
        ["ffmpeg", "-i", "pipe:0", "-f", "s16le", "-ar", str(SAMPLE_RATE), "-ac", str(CHANNELS), "pipe:1"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE
    )

    # Передаем данные с сервера в ffmpeg
    for chunk in r.iter_content(4096):
        if chunk:
            ffmpeg.stdin.write(chunk)
    ffmpeg.stdin.close()

    # Читаем PCM и сразу на колонки
    while True:
        pcm_chunk = ffmpeg.stdout.read(512)
        if not pcm_chunk:
            break
        audio = np.frombuffer(pcm_chunk, dtype=np.int16)
        stream.write(audio)

stream.stop()
stream.close()

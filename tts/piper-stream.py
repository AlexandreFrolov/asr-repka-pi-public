import subprocess
import sounddevice as sd

PIPER_BIN = "piper"
MODEL = "/root/piper-voices/ru/ru_RU-irina-medium.onnx"

SAMPLE_RATE = 22050
CHANNELS = 1
DTYPE = "int16"
BLOCKSIZE = 1024  # ✔ корректный аудиоблок

def speak(text: str):
    proc = subprocess.Popen(
        [
            PIPER_BIN,
            "--model", MODEL,
            "--output_raw"
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL
    )

    proc.stdin.write(text.encode("utf-8"))
    proc.stdin.close()

    with sd.RawOutputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype=DTYPE,
        blocksize=BLOCKSIZE
    ) as stream:

        stream.start()

        while True:
            data = proc.stdout.read(BLOCKSIZE * 2)  # int16 = 2 байта
            if not data:
                break
            stream.write(data)

    proc.wait()

if __name__ == "__main__":
    speak("Привет, Репка Пи. Проверка потокового синтеза.")


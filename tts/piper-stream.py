#!/usr/bin/env python3
import subprocess
import sounddevice as sd
import sys
from pathlib import Path

PIPER_BIN = "piper"
MODEL = "/root/piper-voices/ru/ru_RU-irina-medium.onnx"

SAMPLE_RATE = 22050
CHANNELS = 1
DTYPE = "int16"

BLOCKSIZE = 2048               # ⬅ больше буфер
BYTES_PER_SAMPLE = 2

def speak_from_file(text_path: Path):
    if not text_path.exists():
        print(f"❌ Файл не найден: {text_path}", file=sys.stderr)
        sys.exit(1)

    text = text_path.read_text(encoding="utf-8").strip()
    if not text:
        print("❌ Файл пустой", file=sys.stderr)
        sys.exit(1)

    # 🔊 Явно используем PulseAudio
    sd.default.device = "pulse"
    sd.default.samplerate = SAMPLE_RATE
    sd.default.channels = CHANNELS

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
        blocksize=BLOCKSIZE,
        latency="high"           # ⬅ КЛЮЧЕВО
    ) as stream:

        stream.start()

        while True:
            data = proc.stdout.read(BLOCKSIZE * BYTES_PER_SAMPLE * 2)
            if not data:
                break

            # выравнивание под int16
            if len(data) % BYTES_PER_SAMPLE:
                data = data[:-(len(data) % BYTES_PER_SAMPLE)]

            stream.write(data)

    proc.wait()


def main():
    if len(sys.argv) != 2:
        print(
            f"Использование:\n"
            f"  {sys.argv[0]} <путь_к_текстовому_файлу>",
            file=sys.stderr
        )
        sys.exit(1)

    speak_from_file(Path(sys.argv[1]))


if __name__ == "__main__":
    main()

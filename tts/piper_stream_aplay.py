#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path

PIPER_BIN = "piper"
MODEL_PATH = "/root/piper-voices/ru/ru_RU-irina-medium.onnx"

SAMPLE_RATE = "22050"
FORMAT = "S16_LE"
DEVICE = "pulse"


def speak_from_file(text_path: Path):
    if not text_path.exists():
        print(f"❌ Файл не найден: {text_path}", file=sys.stderr)
        sys.exit(1)

    text = text_path.read_text(encoding="utf-8").strip()
    if not text:
        print("❌ Файл пустой", file=sys.stderr)
        sys.exit(1)

    piper = subprocess.Popen(
        [
            PIPER_BIN,
            "--model", MODEL_PATH,
            "--output_raw"
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL
    )

    aplay = subprocess.Popen(
        [
            "aplay",
            "-r", SAMPLE_RATE,
            "-f", FORMAT,
            "-t", "raw",
            "-D", DEVICE
        ],
        stdin=piper.stdout,
        stderr=subprocess.DEVNULL
    )

    # Передаём текст Piper
    piper.stdin.write(text.encode("utf-8"))
    piper.stdin.close()

    aplay.wait()
    piper.wait()


def main():
    if len(sys.argv) != 2:
        print(
            f"Использование:\n"
            f"  {sys.argv[0]} <путь_к_текстовому_файлу>",
            file=sys.stderr
        )
        sys.exit(1)

    text_file = Path(sys.argv[1])
    speak_from_file(text_file)


if __name__ == "__main__":
    main()


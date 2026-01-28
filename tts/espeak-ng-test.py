import subprocess

def say(text, voice="ru", speed=150, pitch=50):
    """
    Произносит текст с помощью espeak-ng
    
    Параметры:
        voice  — голос/язык (ru, en, de, fr, ro и т.д.)
        speed  — скорость (обычно 80–200)
        pitch  — высота голоса (0–99)
    """
    try:
        subprocess.run([
            "espeak-ng",
            "-v", voice,
            "-s", str(speed),
            "-p", str(pitch),
            text
        ], check=True, capture_output=True)
    except FileNotFoundError:
        print("espeak-ng не найден. Установите его:")
        print("  Linux:   sudo apt install espeak-ng")
        print("  macOS:   brew install espeak")
        print("  Windows: https://github.com/espeak-ng/espeak-ng/releases")
    except subprocess.CalledProcessError as e:
        print("Ошибка espeak-ng:", e.stderr.decode())


# Примеры использования
if __name__ == "__main__":
    say("Привет! Это проверка русского голоса.", voice="ru", speed=140)
    say("Hello, this is a test in English", voice="en-us", speed=160, pitch=60)
    say("Тестируем другой русский голос", voice="ru+f3", speed=135)

import os
import queue
import sys
import json
import sounddevice as sd
from vosk import Model, KaldiRecognizer

# Настройки
MODEL_PATH = "model"
SAMPLE_RATE = 16000

# Очередь для аудиоданных
audio_queue = queue.Queue()

def callback(indata, frames, time, status):
    """Функция обратного вызова для захвата аудио"""
    if status:
        print(f"Ошибка захвата: {status}", file=sys.stderr)
    audio_queue.put(bytes(indata))

def find_usb_microphone():
    """Поиск ID USB-микрофона"""
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if 'USB' in dev['name'] and dev['max_input_channels'] > 0:
            print(f" Найдено устройство: {dev['name']} (ID: {i})")
            return i
    return None

def set_terminal_no_wrap(enable=True):
    """Отключает или включает автоматический перенос строк в терминале"""
    if enable:
        sys.stdout.write("\033[?7l")  # Отключить перенос (DECAWM)
    else:
        sys.stdout.write("\033[?7h")  # Включить перенос (DECAWM)
    sys.stdout.flush()

# Проверка модели
if not os.path.exists(MODEL_PATH):
    print(f"Ошибка: Папка '{MODEL_PATH}' не найдена.")
    exit(1)

# Инициализация Vosk
model = Model(MODEL_PATH)
rec = KaldiRecognizer(model, SAMPLE_RATE)

device_id = find_usb_microphone()
if device_id is None:
    print("USB-микрофон не найден.")
    exit(1)

print("-" * 30)
print("Микрофон готов. Говорите...")
print("-" * 30)

try:
    # Отключаем перенос строк, чтобы длинные фразы не плодили новые строки
    set_terminal_no_wrap(True)

    with sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=8000, device=device_id,
                            dtype='int16', channels=1, callback=callback):
        
        while True:
            data = audio_queue.get()
            if rec.AcceptWaveform(data):
                # Очищаем текущую строку перед выводом финального результата
                # \r - в начало, \033[K - очистить до конца строки
                sys.stdout.write("\r\033[K")
                
                result = json.loads(rec.Result())
                text = result.get("text", "")
                if text:
                    # Печатаем результат и переходим на новую строку
                    sys.stdout.write(f"Результат: {text}\n")
                    sys.stdout.flush()
            else:
                # Промежуточный результат
                partial = json.loads(rec.PartialResult())
                partial_text = partial.get("partial", "")
                if partial_text:
                    # \r возвращает в начало, текст пишется поверх старого, 
                    # \033[K убирает лишние символы справа, если новая фраза короче старой
                    sys.stdout.write(f"\r {partial_text}...\033[K")
                    sys.stdout.flush()

except KeyboardInterrupt:
    # Возвращаем терминал в нормальное состояние
    set_terminal_no_wrap(False)
    print("\n\nПрограмма остановлена пользователем.")
except Exception as e:
    set_terminal_no_wrap(False)
    print(f"\nПроизошла ошибка: {e}")
finally:
    # На всякий случай включаем перенос обратно
    set_terminal_no_wrap(False)
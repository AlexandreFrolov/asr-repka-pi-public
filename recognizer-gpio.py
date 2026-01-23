# -*- coding: utf-8 -*-

import os
import queue
import sys
import json
import sounddevice as sd
from vosk import Model, KaldiRecognizer
import RepkaPi.GPIO as GPIO 
from time import sleep

# --- Настройки ---
MODEL_PATH = "model"
SAMPLE_RATE = 16000
LED_PIN = 7  # Пин светодиода (Board numbering)

# --- Настройка GPIO ---
GPIO.setmode(GPIO.BOARD)
GPIO.setup(LED_PIN, GPIO.OUT)
# Изначально светодиод выключен
GPIO.output(LED_PIN, GPIO.LOW)

audio_queue = queue.Queue()

def callback(indata, frames, time, status):
    if status:
        print(f"Ошибка захвата: {status}", file=sys.stderr)
    audio_queue.put(bytes(indata))

def find_usb_microphone():
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if 'USB' in dev['name'] and dev['max_input_channels'] > 0:
            print(f" Найдено устройство: {dev['name']} (ID: {i})")
            return i
    return None

def set_terminal_no_wrap(enable=True):
    """Устраняет дублирование строк на консоли Repka Pi"""
    if enable:
        sys.stdout.write("\033[?7l") 
    else:
        sys.stdout.write("\033[?7h") 
    sys.stdout.flush()

if not os.path.exists(MODEL_PATH):
    print(f"Ошибка: Папка '{MODEL_PATH}' не найдена.")
    GPIO.cleanup()
    exit(1)

model = Model(MODEL_PATH)
rec = KaldiRecognizer(model, SAMPLE_RATE)

device_id = find_usb_microphone()
if device_id is None:
    print("USB-микрофон не найден.")
    GPIO.cleanup()
    exit(1)

print("-" * 30)
print("Система готова.")
print("Команды: 'лампа' - включить, 'погасить' - выключить.")
print("-" * 30)

try:
    set_terminal_no_wrap(True)

    with sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=8000, device=device_id,
                            dtype='int16', channels=1, callback=callback):
        
        while True:
            data = audio_queue.get()
            if rec.AcceptWaveform(data):
                # Очищаем строку перед выводом результата
                sys.stdout.write("\r\033[K")
                
                result_json = json.loads(rec.Result())
                text = result_json.get("text", "").lower() # Переводим в нижний регистр для надежности
                
                if text:
                    print(f"Результат: {text}")
                    
                    # --- Логика управления командами ---
                    if "лампа" in text:
                        print(">>> Исполняю: ВКЛЮЧИТЬ СВЕТ")
                        GPIO.output(LED_PIN, GPIO.HIGH)
                    
                    elif "погасить" in text:
                        print(">>> Исполняю: ВЫКЛЮЧИТЬ СВЕТ")
                        GPIO.output(LED_PIN, GPIO.LOW)
            else:
                # Промежуточный результат (динамическое отображение)
                partial = json.loads(rec.PartialResult())
                partial_text = partial.get("partial", "")
                if partial_text:
                    sys.stdout.write(f"\r Слушаю: {partial_text}...\033[K")
                    sys.stdout.flush()

except KeyboardInterrupt:
    print("\nОстановка программы...")
finally:
    set_terminal_no_wrap(False)
    GPIO.output(LED_PIN, GPIO.LOW)
    GPIO.cleanup()
    print("Настройки сброшены. До свидания!")

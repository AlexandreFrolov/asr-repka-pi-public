import time
import numpy as np
import sounddevice as sd
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from piper import PiperVoice
import warnings
import threading
import queue

# Игнорируем предупреждения от sounddevice
warnings.filterwarnings("ignore", message="Exception ignored from cffi callback")

# Настройки модели
MODEL_PATH = "ru_RU-irina-medium.onnx"
CONFIG_PATH = "ru_RU-irina-medium.onnx.json"

# Инициализация модели
voice = PiperVoice.load(MODEL_PATH, config_path=CONFIG_PATH)

app = FastAPI()

# Глобальные переменные для аудиопотока
audio_stream = None
audio_queue = queue.Queue()
is_playing = False

class TTSRequest(BaseModel):
    text: str

def init_audio_stream():
    """Инициализация аудиопотока при запуске"""
    global audio_stream
    try:
        if audio_stream is None or not audio_stream.active:
            samplerate = voice.config.sample_rate
            print(f"Инициализация аудиопотока с samplerate={samplerate}")
            
            # Создаем новый поток с настройками для стабильности
            audio_stream = sd.OutputStream(
                samplerate=samplerate, 
                channels=1,
                dtype='int16',
                blocksize=2048,  # Размер буфера
                latency='high'   # Высокая задержка для стабильности
            )
            audio_stream.start()
            
            # Воспроизводим тестовый тон для инициализации
            test_duration = 0.05  # 50 мс
            test_samples = int(test_duration * samplerate)
            test_tone = (0.01 * np.sin(2 * np.pi * 440 * np.linspace(0, test_duration, test_samples, False)) * 32767).astype(np.int16)
            audio_stream.write(test_tone)
            print(f"Аудиопоток инициализирован успешно")
    except Exception as e:
        print(f"Ошибка при инициализации аудиопотока: {e}")
        audio_stream = None

def audio_worker():
    """Фоновый рабочий поток для обработки аудио"""
    global is_playing
    
    while True:
        try:
            # Ждем следующую фразу для воспроизведения
            text = audio_queue.get(timeout=1)
            
            if text is None:  # Сигнал завершения
                break
                
            is_playing = True
            synthesize_and_play(text)
            is_playing = False
            audio_queue.task_done()
            
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Ошибка в аудио-воркере: {e}")
            is_playing = False

def synthesize_and_play(text: str):
    """Синтез и воспроизведение одной фразы"""
    global audio_stream
    
    try:
        samplerate = voice.config.sample_rate
        
        # Проверяем и переинициализируем поток если нужно
        if audio_stream is None or not audio_stream.active:
            init_audio_stream()
            if audio_stream is None:
                print("Не удалось инициализировать аудиопоток")
                return
        
        # Синтезируем аудио
        audio_chunks = []
        for audio_chunk in voice.synthesize(text):
            audio_chunks.append(audio_chunk.audio_int16_array)
        
        if not audio_chunks:
            return
        
        # Объединяем чанки
        audio_data = np.concatenate(audio_chunks)
        
        # Добавляем 30 мс тишины в начало для предотвращения проглатывания
        silence_duration = 0.03  # 30 мс
        silence_samples = int(silence_duration * samplerate)
        
        if silence_samples > 0:
            silence = np.zeros(silence_samples, dtype=np.int16)
            audio_data = np.concatenate([silence, audio_data])
        
        # Добавляем 10 мс тишины в конец для предотвращения обрезания
        silence_end_samples = int(0.01 * samplerate)  # 10 мс
        if silence_end_samples > 0:
            silence_end = np.zeros(silence_end_samples, dtype=np.int16)
            audio_data = np.concatenate([audio_data, silence_end])
        
        print(f"Воспроизведение: '{text}'")
        print(f"Размер аудио данных: {len(audio_data)} samples")
        print(f"Частота дискретизации: {samplerate} Hz")
        
        # Воспроизводим через уже инициализированный поток
        try:
            # Разбиваем на чанки для более плавного воспроизведения
            chunk_size = 4096
            total_samples = len(audio_data)
            
            for i in range(0, total_samples, chunk_size):
                end_idx = min(i + chunk_size, total_samples)
                chunk = audio_data[i:end_idx]
                
                # Дополняем последний чанк если нужно
                if len(chunk) < chunk_size:
                    padding = np.zeros(chunk_size - len(chunk), dtype=np.int16)
                    chunk = np.concatenate([chunk, padding])
                
                audio_stream.write(chunk)
                
        except Exception as e:
            print(f"Ошибка при записи в аудиопоток: {e}")
            # Попробуем переинициализировать поток
            init_audio_stream()
            
    except Exception as e:
        print(f"Ошибка при синтезе или воспроизведении: {e}")
        import traceback
        traceback.print_exc()

def play_audio(text: str):
    """Добавление текста в очередь на воспроизведение"""
    # Проверяем и инициализируем поток если нужно
    global audio_stream
    if audio_stream is None:
        init_audio_stream()
    
    # Добавляем текст в очередь
    audio_queue.put(text)

@app.post("/say")
async def say_text(request: TTSRequest, background_tasks: BackgroundTasks):
    """Обработка запроса на синтез речи"""
    # Добавляем задачу в фоновые задачи
    background_tasks.add_task(play_audio, request.text)
    return {"status": "processing", "text": request.text}

@app.on_event("startup")
async def startup_event():
    """Инициализация при запуске сервера"""
    print("Инициализация сервера TTS...")
    
    # Инициализируем аудиопоток
    init_audio_stream()
    
    # Запускаем фоновый рабочий поток для воспроизведения
    audio_thread = threading.Thread(target=audio_worker, daemon=True)
    audio_thread.start()
    
    print(f"Сервер TTS запущен. Частота дискретизации: {voice.config.sample_rate} Hz")

@app.on_event("shutdown")
async def shutdown_event():
    """Очистка при завершении сервера"""
    print("Завершение работы сервера TTS...")
    
    # Останавливаем аудиопоток
    global audio_stream
    if audio_stream is not None:
        try:
            audio_stream.stop()
            audio_stream.close()
        except:
            pass
    
    # Отправляем сигнал завершения воркеру
    try:
        audio_queue.put(None)
    except:
        pass
    
    print("Сервер TTS остановлен")

if __name__ == "__main__":
    import uvicorn
    print(f"Запуск сервера TTS на порту 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
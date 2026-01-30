import time
import numpy as np
import sounddevice as sd
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from piper import PiperVoice
import warnings
import threading
import queue
from typing import Optional

# Игнорируем предупреждения от sounddevice
warnings.filterwarnings("ignore", message="Exception ignored from cffi callback")

# Настройки модели
MODEL_PATH = "ru_RU-irina-medium.onnx"
CONFIG_PATH = "ru_RU-irina-medium.onnx.json"

# Инициализация модели
voice = PiperVoice.load(MODEL_PATH, config_path=CONFIG_PATH)

app = FastAPI()

# Глобальные переменные для аудиопотока
audio_stream: Optional[sd.OutputStream] = None
audio_queue = queue.Queue()
is_playing = False
stop_worker = False
samplerate = voice.config.sample_rate

class TTSRequest(BaseModel):
    text: str

def init_audio_stream():
    """Инициализация аудиопотока при запуске"""
    global audio_stream, samplerate
    
    try:
        if audio_stream is not None:
            try:
                audio_stream.stop()
                audio_stream.close()
            except:
                pass
        
        print(f"Инициализация аудиопотока с samplerate={samplerate}")
        
        # Увеличиваем буферы для предотвращения underrun
        audio_stream = sd.OutputStream(
            samplerate=samplerate, 
            channels=1,
            dtype='int16',
            blocksize=4096,  # Увеличиваем размер буфера
            latency='high',
            device=None,  # Используем устройство по умолчанию
            extra_settings=None
        )
        audio_stream.start()
        
        print(f"Аудиопоток инициализирован успешно")
    except Exception as e:
        print(f"Ошибка при инициализации аудиопотока: {e}")
        audio_stream = None

def audio_worker():
    """Фоновый рабочий поток для обработки аудио"""
    global is_playing, stop_worker
    
    while not stop_worker:
        try:
            # Ждем следующую фразу для воспроизведения
            text = audio_queue.get(timeout=0.1)
            
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
            import traceback
            traceback.print_exc()
            is_playing = False

def synthesize_and_play(text: str):
    """Синтез и воспроизведение одной фразы"""
    global audio_stream, samplerate
    
    try:
        # Нормализация текста перед синтезом
        normalized_text = normalize_text(text)
        
        # Проверяем и переинициализируем поток если нужно
        if audio_stream is None or not audio_stream.active:
            init_audio_stream()
            if audio_stream is None:
                print("Не удалось инициализировать аудиопоток")
                return
        
        # Синтезируем аудио
        audio_chunks = []
        for audio_chunk in voice.synthesize(normalized_text):
            audio_chunks.append(audio_chunk.audio_int16_array)
        
        if not audio_chunks:
            return
        
        # Объединяем чанки
        audio_data = np.concatenate(audio_chunks)
        
        # Добавляем тишину для предотвращения обрезания
        silence_samples = int(0.02 * samplerate)  # 20 мс тишины
        
        if silence_samples > 0:
            silence = np.zeros(silence_samples, dtype=np.int16)
            audio_data = np.concatenate([silence, audio_data, silence])
        
        print(f"Воспроизведение: '{text}'")
        print(f"Размер аудио данных: {len(audio_data)} samples")
        print(f"Частота дискретизации: {samplerate} Hz")
        
        # Разбиваем на чанки для плавного воспроизведения
        chunk_size = 4096
        total_samples = len(audio_data)
        
        for i in range(0, total_samples, chunk_size):
            if audio_stream is None or not audio_stream.active:
                break
                
            end_idx = min(i + chunk_size, total_samples)
            chunk = audio_data[i:end_idx]
            
            # Дополняем последний чанк если нужно
            if len(chunk) < chunk_size:
                padding = np.zeros(chunk_size - len(chunk), dtype=np.int16)
                chunk = np.concatenate([chunk, padding])
            
            try:
                audio_stream.write(chunk)
            except Exception as e:
                print(f"Ошибка при записи в аудиопоток: {e}")
                # Переинициализируем поток
                init_audio_stream()
                # Пропускаем этот чанк
                continue
        
    except Exception as e:
        print(f"Ошибка при синтезе или воспроизведении: {e}")
        import traceback
        traceback.print_exc()

def normalize_text(text: str) -> str:
    """Нормализация текста для лучшего синтеза"""
    import re
    
    # Заменяем длинное тире на короткое
    text = text.replace('—', '-').replace('–', '-')
    
    # Убираем лишние пробелы и переносы строк
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Добавляем пробелы после знаков препинания, если их нет
    text = re.sub(r'([.,!?:;])(?=[^\s])', r'\1 ', text)
    
    # Особый случай для "Привет!" - убедимся, что он правильно интерпретируется
    if 'Привет!' in text:
        # Заменяем восклицательный знак после слова, если нужно
        text = text.replace('Привет!', 'Привет !')
    
    return text

def play_audio(text: str):
    """Добавление текста в очередь на воспроизведение"""
    # Проверяем и инициализируем поток если нужно
    global audio_stream
    if audio_stream is None or not audio_stream.active:
        init_audio_stream()
    
    # Добавляем текст в очередь, если он не пустой
    if text and text.strip():
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
    
    print(f"Сервер TTS запущен. Частота дискретизации: {samplerate} Hz")
    print(f"Используется модель: {MODEL_PATH}")

@app.on_event("shutdown")
async def shutdown_event():
    """Очистка при завершении сервера"""
    print("Завершение работы сервера TTS...")
    
    global stop_worker, audio_stream
    
    # Останавливаем фоновый поток
    stop_worker = True
    
    # Останавливаем аудиопоток
    if audio_stream is not None:
        try:
            audio_stream.stop()
            audio_stream.close()
        except:
            pass
        audio_stream = None
    
    print("Сервер TTS остановлен")

if __name__ == "__main__":
    import uvicorn
    print(f"Запуск сервера TTS на порту 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
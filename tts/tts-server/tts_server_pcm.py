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
import re

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
 
def preprocess_text(text: str) -> str:
    """
    Предобработка текста аналогично тому, как это делает Piper CLI.
    Основная идея: передавать текст построчно, как при чтении из файла.
    """
    # 1. Разбиваем текст на строки (как при чтении из файла)
    lines = text.split('\n')
    
    # 2. Убираем пустые строки и пробелы в начале/конце
    lines = [line.strip() for line in lines if line.strip()]
    
    # 3. Возвращаем обратно, объединяя символом новой строки
    #    Это важно: Piper CLI обрабатывает каждую строку отдельно
    return '\n'.join(lines)

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
        
        # Увеличиваем блоки и буфер для предотвращения underrun
        audio_stream = sd.OutputStream(
            samplerate=samplerate, 
            channels=1,
            dtype='int16',
            blocksize=8192,  # Увеличили в 2 раза
            latency='high',
            extra_settings=None
        )
        audio_stream.start()
        
        # Заполняем буфер тишиной для предотвращения underrun
        fill_buffer_samples = 8192  # Заполняем полный блок
        silence = np.zeros(fill_buffer_samples, dtype=np.int16)
        audio_stream.write(silence)
        
        print(f"Аудиопоток инициализирован успешно с blocksize=8192")
    except Exception as e:
        print(f"Ошибка при инициализации аудиопотока: {e}")
        audio_stream = None

def audio_worker():
    """Фоновый рабочий поток для обработки аудио"""
    global is_playing, stop_worker
    
    while not stop_worker:
        try:
            text = audio_queue.get(timeout=0.1)
            
            if text is None:
                break
                
            is_playing = True
            synthesize_and_play_line_by_line(text)
            is_playing = False
            audio_queue.task_done()
            
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Ошибка в аудио-воркере: {e}")
            import traceback
            traceback.print_exc()
            is_playing = False

def synthesize_and_play_line_by_line(text: str):
    """Синтез и воспроизведение построчно (как в Piper CLI)"""
    global audio_stream, samplerate
    
    try:
        # Предобрабатываем текст
        processed_text = preprocess_text(text)
        
        # Разбиваем на строки
        lines = processed_text.split('\n')
        
        if not lines:
            return
        
        # Проверяем аудиопоток
        if audio_stream is None or not audio_stream.active:
            init_audio_stream()
            if audio_stream is None:
                print("Не удалось инициализировать аудиопоток")
                return
        
        # Синтезируем и воспроизводим каждую строку отдельно
        all_audio_chunks = []
        
        for line_num, line in enumerate(lines):
            print(f"Синтез строки {line_num + 1}: '{line}'")
            
            # Синтезируем текущую строку
            line_audio_chunks = []
            for audio_chunk in voice.synthesize(line):
                line_audio_chunks.append(audio_chunk.audio_int16_array)
            
            if not line_audio_chunks:
                continue
            
            # Объединяем чанки для этой строки
            line_audio = np.concatenate(line_audio_chunks)
            
            # Добавляем паузу между строками (кроме последней)
            if line_num < len(lines) - 1:
                pause_samples = int(0.05 * samplerate)  # 50 мс паузы
                pause = np.zeros(pause_samples, dtype=np.int16)
                line_audio = np.concatenate([line_audio, pause])
            
            all_audio_chunks.append(line_audio)
        
        if not all_audio_chunks:
            return
        
        # Объединяем все строки
        audio_data = np.concatenate(all_audio_chunks)
        
        # Добавляем небольшие паузы в начале и конце
        silence_samples = int(0.02 * samplerate)
        silence = np.zeros(silence_samples, dtype=np.int16)
        audio_data = np.concatenate([silence, audio_data, silence])
        
        print(f"Общий размер аудио: {len(audio_data)} samples")
        
        # Воспроизводим с небольшими чанками для плавности
        try:
            block_size = 2048  # Уменьшенный размер блока для плавности
            total_samples = len(audio_data)
            
            for i in range(0, total_samples, block_size):
                if audio_stream is None or not audio_stream.active:
                    break
                    
                end_idx = min(i + block_size, total_samples)
                chunk = audio_data[i:end_idx]
                
                # Дополняем последний блок
                if len(chunk) < block_size:
                    padding = np.zeros(block_size - len(chunk), dtype=np.int16)
                    chunk = np.concatenate([chunk, padding])
                
                audio_stream.write(chunk)
                
        except Exception as e:
            print(f"Ошибка при воспроизведении: {e}")
            init_audio_stream()
            
    except Exception as e:
        print(f"Ошибка при синтезе: {e}")
        import traceback
        traceback.print_exc()

def play_audio(text: str):
    """Добавление текста в очередь на воспроизведение"""
    global audio_stream
    if audio_stream is None or not audio_stream.active:
        init_audio_stream()
    
    if text and text.strip():
        audio_queue.put(text)

@app.post("/say")
async def say_text(request: TTSRequest, background_tasks: BackgroundTasks):
    """Обработка запроса на синтез речи"""
    background_tasks.add_task(play_audio, request.text)
    return {"status": "processing", "text": request.text}

@app.on_event("startup")
async def startup_event():
    """Инициализация при запуске сервера"""
    print("Инициализация сервера TTS...")
    
    init_audio_stream()
    
    audio_thread = threading.Thread(target=audio_worker, daemon=True)
    audio_thread.start()
    
    print(f"Сервер TTS запущен. Частота дискретизации: {samplerate} Hz")

@app.on_event("shutdown")
async def shutdown_event():
    """Очистка при завершении сервера"""
    print("Завершение работы сервера TTS...")
    
    global stop_worker, audio_stream
    stop_worker = True
    
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
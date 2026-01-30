import time
import numpy as np
import sounddevice as sd
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from piper import PiperVoice
import warnings
import threading
import queue
from typing import Optional, Dict
import re
import hashlib

# Игнорируем предупреждения от sounddevice
warnings.filterwarnings("ignore", message="Exception ignored from cffi callback")

# Настройки модели
MODEL_PATH = "ru_RU-irina-medium.onnx"
CONFIG_PATH = "ru_RU-irina-medium.onnx.json"

# Инициализация модели
voice = PiperVoice.load(MODEL_PATH, config_path=CONFIG_PATH)

app = FastAPI()

# Глобальные переменные для аудиопотока
audio_queue = queue.Queue()
samplerate = voice.config.sample_rate
is_playing = False
stop_worker = False

# Кэш для аудиоданных
audio_cache: Dict[str, np.ndarray] = {}
CACHE_MAX_SIZE = 100  # Максимальное количество записей в кэше

# Настройки для предотвращения underrun
BUFFER_SIZE = 4096
CHUNK_SIZE = 2048
PRE_BUFFER_MS = 100

class TTSRequest(BaseModel):
    text: str

def get_text_hash(text: str) -> str:
    """Вычисление хэша текста для использования в качестве ключа кэша."""
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def add_to_cache(text: str, audio_data: np.ndarray):
    """Добавление аудиоданных в кэш."""
    text_hash = get_text_hash(text)
    if len(audio_cache) >= CACHE_MAX_SIZE:
        # Удаляем первый элемент (можно улучшить, используя LRU, но для простоты удаляем случайный)
        audio_cache.popitem()
    audio_cache[text_hash] = audio_data

def get_from_cache(text: str) -> Optional[np.ndarray]:
    """Получение аудиоданных из кэша."""
    text_hash = get_text_hash(text)
    return audio_cache.get(text_hash)

class AudioPlayerThread(threading.Thread):
    """Поток-плейер для воспроизведения аудио"""
    def __init__(self, audio_data, samplerate, player_id=0):
        threading.Thread.__init__(self)
        self.audio_data = audio_data
        self.samplerate = samplerate
        self.player_id = player_id
        self.stream = None
        self.daemon = True
        self.audio_duration = len(audio_data) / samplerate if len(audio_data) > 0 else 0
    
    def init_stream(self):
        """Инициализация аудиопотока для этого плеера"""
        try:
            if self.stream is not None:
                try:
                    self.stream.stop()
                    self.stream.close()
                except:
                    pass
            
            self.stream = sd.OutputStream(
                samplerate=self.samplerate,
                channels=1,
                dtype='int16',
                blocksize=BUFFER_SIZE,
                latency='high'
            )
            self.stream.start()
            
            pre_buffer_samples = int((PRE_BUFFER_MS / 1000.0) * self.samplerate)
            silence = np.zeros(pre_buffer_samples, dtype=np.int16)
            self.stream.write(silence)
            time.sleep(pre_buffer_samples / self.samplerate + 0.01)
            
            print(f"Плеер {self.player_id}: Аудиопоток инициализирован, предбуфер {pre_buffer_samples} samples")
            return True
            
        except Exception as e:
            print(f"Плеер {self.player_id}: Ошибка при инициализации аудиопотока: {e}")
            self.stream = None
            return False
    
    def run(self):
        """Запуск воспроизведения в отдельном потоке"""
        global is_playing
        
        try:
            is_playing = True
            
            if not self.init_stream():
                print(f"Плеер {self.player_id}: Не удалось инициализировать аудиопоток")
                is_playing = False
                return
            
            if self.stream is None:
                print(f"Плеер {self.player_id}: Аудиопоток не создан")
                is_playing = False
                return
            
            audio_data = self.audio_data
            total_samples = len(audio_data)
            
            if total_samples == 0:
                print(f"Плеер {self.player_id}: Нет аудиоданных для воспроизведения")
                is_playing = False
                return
            
            print(f"Плеер {self.player_id}: Начало воспроизведения, {total_samples} samples, длительность: {self.audio_duration:.2f} сек")
            
            try:
                self.stream.write(audio_data)
                estimated_play_time = total_samples / self.samplerate
                time.sleep(estimated_play_time + 0.05)
            except Exception as e:
                print(f"Плеер {self.player_id}: Ошибка при записи аудио: {e}")
                if self.stream and self.stream.active:
                    chunk_size = CHUNK_SIZE
                    for i in range(0, total_samples, chunk_size):
                        chunk_end = min(i + chunk_size, total_samples)
                        chunk = audio_data[i:chunk_end]
                        
                        if len(chunk) < chunk_size:
                            padding = np.zeros(chunk_size - len(chunk), dtype=np.int16)
                            chunk = np.concatenate([chunk, padding])
                        
                        self.stream.write(chunk)
            
            print(f"Плеер {self.player_id}: Воспроизведение завершено")
            
        except Exception as e:
            print(f"Плеер {self.player_id}: Ошибка при воспроизведении: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            if self.stream is not None:
                try:
                    time.sleep(0.02)
                    self.stream.stop()
                    self.stream.close()
                except:
                    pass
                self.stream = None
            
            is_playing = False

def preprocess_text(text: str) -> str:
    lines = text.split('\n')
    lines = [line.strip() for line in lines if line.strip()]
    return '\n'.join(lines)

def synthesize_text(text: str, use_cache: bool = True) -> Optional[np.ndarray]:
    """Синтез текста в аудиоданные с возможностью использования кэша."""
    # Если используем кэш и текст уже есть в кэше, возвращаем из кэша
    if use_cache:
        cached_audio = get_from_cache(text)
        if cached_audio is not None:
            print(f"Аудио для текста найдено в кэше, размер: {len(cached_audio)} samples")
            return cached_audio.copy()  # Возвращаем копию, чтобы оригинал в кэше не менялся
    
    try:
        processed_text = preprocess_text(text)
        lines = processed_text.split('\n')
        
        if not lines:
            return None
        
        all_audio_chunks = []
        
        for line_num, line in enumerate(lines):
            print(f"Синтез строки {line_num + 1}: '{line}'")
            
            line_audio_chunks = []
            for audio_chunk in voice.synthesize(line):
                line_audio_chunks.append(audio_chunk.audio_int16_array)
            
            if not line_audio_chunks:
                continue
            
            line_audio = np.concatenate(line_audio_chunks)
            
            if line_num < len(lines) - 1:
                pause_samples = int(0.05 * samplerate)
                pause = np.zeros(pause_samples, dtype=np.int16)
                line_audio = np.concatenate([line_audio, pause])
            
            all_audio_chunks.append(line_audio)
        
        if not all_audio_chunks:
            return None
        
        audio_data = np.concatenate(all_audio_chunks)
        
        initial_silence = int(0.1 * samplerate)
        end_silence = int(0.05 * samplerate)
        
        silence_start = np.zeros(initial_silence, dtype=np.int16)
        silence_end = np.zeros(end_silence, dtype=np.int16)
        
        test_tone_samples = int(0.005 * samplerate)
        if test_tone_samples > 0:
            test_tone = (0.01 * np.sin(2 * np.pi * 1000 * np.linspace(0, 0.005, test_tone_samples)) * 32767).astype(np.int16)
            audio_data = np.concatenate([silence_start, test_tone, audio_data, silence_end])
        else:
            audio_data = np.concatenate([silence_start, audio_data, silence_end])
        
        print(f"Синтез завершен: {len(audio_data)} samples, начальная пауза: {initial_silence} samples ({initial_silence/samplerate*1000:.1f} мс)")
        
        # Сохраняем в кэш
        if use_cache:
            add_to_cache(text, audio_data.copy())  # Сохраняем копию
        
        return audio_data
        
    except Exception as e:
        print(f"Ошибка при синтезе: {e}")
        import traceback
        traceback.print_exc()
        return None

def audio_worker():
    global is_playing, stop_worker
    player_counter = 0
    
    while not stop_worker:
        try:
            text = audio_queue.get(timeout=0.1)
            
            if text is None:
                break
            
            if is_playing:
                print(f"Ожидание завершения текущего воспроизведения...")
                while is_playing and not stop_worker:
                    time.sleep(0.1)
                time.sleep(0.2)
            
            print(f"Синтез текста: '{text[:50]}...'")
            audio_data = synthesize_text(text, use_cache=True)
            
            if audio_data is not None and len(audio_data) > 0:
                player_counter += 1
                player = AudioPlayerThread(audio_data, samplerate, player_counter)
                player.start()
                time.sleep(0.05)
            
            audio_queue.task_done()
            
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Ошибка в аудио-воркере: {e}")
            import traceback
            traceback.print_exc()

def play_audio(text: str):
    if text and text.strip():
        log_text = text[:100] + "..." if len(text) > 100 else text
        print(f"Добавление в очередь: '{log_text}'")
        audio_queue.put(text)

@app.post("/say")
async def say_text(request: TTSRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(play_audio, request.text)
    return {"status": "processing", "text": request.text}

@app.get("/status")
async def get_status():
    return {
        "status": "running",
        "is_playing": is_playing,
        "queue_size": audio_queue.qsize(),
        "cache_size": len(audio_cache),
        "samplerate": samplerate
    }

@app.post("/clear_cache")
async def clear_cache():
    global audio_cache
    audio_cache.clear()
    return {"status": "cache_cleared"}

@app.on_event("startup")
async def startup_event():
    print("Инициализация сервера TTS...")
    print(f"Частота дискретизации: {samplerate} Hz")
    print(f"Настройки: буфер={BUFFER_SIZE}, чанк={CHUNK_SIZE}, предбуфер={PRE_BUFFER_MS}мс")
    
    audio_thread = threading.Thread(target=audio_worker, daemon=True)
    audio_thread.start()
    
    print("Сервер TTS запущен")

@app.on_event("shutdown")
async def shutdown_event():
    print("Завершение работы сервера TTS...")
    
    global stop_worker
    stop_worker = True
    
    while not audio_queue.empty():
        try:
            audio_queue.get_nowait()
            audio_queue.task_done()
        except:
            pass
    
    time.sleep(1.0)
    
    print("Сервер TTS остановлен")

if __name__ == "__main__":
    import uvicorn
    print(f"Запуск сервера TTS на порту 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
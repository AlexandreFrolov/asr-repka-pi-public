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
from contextlib import asynccontextmanager

# Игнорируем предупреждения от sounddevice
warnings.filterwarnings("ignore", message="Exception ignored from cffi callback")

# Настройки модели
MODEL_PATH = "ru_RU-irina-medium.onnx"
CONFIG_PATH = "ru_RU-irina-medium.onnx.json"

# Инициализация модели
voice = PiperVoice.load(MODEL_PATH, config_path=CONFIG_PATH)

# Глобальные переменные
audio_queue = queue.Queue()
samplerate = voice.config.sample_rate
is_playing = False
stop_worker = False

# Настройки аудио
BUFFER_SIZE = 4096
CHUNK_SIZE = 2048
PRE_BUFFER_MS = 100

class TTSRequest(BaseModel):
    text: str

class AudioPlayerThread(threading.Thread):
    """Поток-плейер для воспроизведения аудио"""
    def __init__(self, audio_data, samplerate, player_id=0):
        threading.Thread.__init__(self)
        self.audio_data = audio_data
        self.samplerate = samplerate
        self.player_id = player_id
        self.stream = None
        self.daemon = True
    
    def init_stream(self):
        """Инициализация аудиопотока"""
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
            
            # Заполняем буфер тишиной
            pre_buffer_samples = int((PRE_BUFFER_MS / 1000.0) * self.samplerate)
            silence = np.zeros(pre_buffer_samples, dtype=np.int16)
            self.stream.write(silence)
            time.sleep(pre_buffer_samples / self.samplerate + 0.01)
            
            return True
            
        except Exception as e:
            print(f"Ошибка при инициализации аудиопотока: {e}")
            self.stream = None
            return False
    
    def run(self):
        """Запуск воспроизведения"""
        global is_playing
        
        try:
            is_playing = True
            
            if not self.init_stream():
                is_playing = False
                return
            
            audio_data = self.audio_data
            total_samples = len(audio_data)
            
            if total_samples == 0:
                is_playing = False
                return
            
            # Воспроизводим аудио
            try:
                self.stream.write(audio_data)
                estimated_play_time = total_samples / self.samplerate
                time.sleep(estimated_play_time + 0.05)
            except Exception as e:
                print(f"Ошибка при воспроизведении: {e}")
            
        except Exception as e:
            print(f"Ошибка в потоке-плейере: {e}")
        
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

def synthesize_text(text: str) -> Optional[np.ndarray]:
    """Синтез текста в аудиоданные"""
    try:
        # Обрабатываем строки
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        if not lines:
            return None
        
        all_audio_chunks = []
        
        for line in lines:
            line_audio_chunks = []
            for audio_chunk in voice.synthesize(line):
                line_audio_chunks.append(audio_chunk.audio_int16_array)
            
            if line_audio_chunks:
                line_audio = np.concatenate(line_audio_chunks)
                all_audio_chunks.append(line_audio)
        
        if not all_audio_chunks:
            return None
        
        audio_data = np.concatenate(all_audio_chunks)
        
        # Добавляем паузы
        initial_silence = int(0.1 * samplerate)
        end_silence = int(0.05 * samplerate)
        
        silence_start = np.zeros(initial_silence, dtype=np.int16)
        silence_end = np.zeros(end_silence, dtype=np.int16)
        
        audio_data = np.concatenate([silence_start, audio_data, silence_end])
        
        return audio_data
        
    except Exception as e:
        print(f"Ошибка при синтезе: {e}")
        return None

def audio_worker():
    """Фоновый рабочий поток"""
    global is_playing, stop_worker
    player_counter = 0
    
    while not stop_worker:
        try:
            text = audio_queue.get(timeout=0.1)
            
            if text is None:
                break
            
            # Ждем завершения текущего воспроизведения
            if is_playing:
                while is_playing and not stop_worker:
                    time.sleep(0.1)
                time.sleep(0.2)
            
            # Синтезируем и воспроизводим
            audio_data = synthesize_text(text)
            
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

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan контекстный менеджер"""
    print("Запуск сервера TTS...")
    
    global audio_thread
    audio_thread = threading.Thread(target=audio_worker, daemon=True)
    audio_thread.start()
    
    yield
    
    print("Остановка сервера TTS...")
    
    global stop_worker
    stop_worker = True
    
    while not audio_queue.empty():
        try:
            audio_queue.get_nowait()
            audio_queue.task_done()
        except:
            pass
    
    time.sleep(1.0)
    print("Сервер остановлен")

# Создаем FastAPI приложение
app = FastAPI(lifespan=lifespan)

@app.post("/say")
async def say_text(request: TTSRequest, background_tasks: BackgroundTasks):
    """Обработка запроса на синтез речи"""
    background_tasks.add_task(lambda: audio_queue.put(request.text))
    return {"status": "processing", "text": request.text}

@app.get("/status")
async def get_status():
    """Получение статуса сервера"""
    return {
        "status": "running",
        "is_playing": is_playing,
        "queue_size": audio_queue.qsize(),
        "samplerate": samplerate
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
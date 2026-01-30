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
from contextlib import asynccontextmanager

# Игнорируем предупреждения от sounddevice
warnings.filterwarnings("ignore", message="Exception ignored from cffi callback")

# Настройки модели
MODEL_PATH = "ru_RU-irina-medium.onnx"
CONFIG_PATH = "ru_RU-irina-medium.onnx.json"

# Инициализация модели
voice = PiperVoice.load(MODEL_PATH, config_path=CONFIG_PATH)

# Глобальные переменные для аудиопотока
audio_queue = queue.Queue()
samplerate = voice.config.sample_rate
is_playing = False
stop_worker = False

# Настройки для предотвращения underrun
BUFFER_SIZE = 4096
CHUNK_SIZE = 2048  # Увеличиваем размер чанка
PRE_BUFFER_MS = 100  # 100 мс предварительного буфера

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
        self.audio_duration = len(audio_data) / samplerate if len(audio_data) > 0 else 0
    
    def init_stream(self):
        """Инициализация аудиопотока для этого плеера"""
        try:
            # Закрываем старый поток, если есть
            if self.stream is not None:
                try:
                    self.stream.stop()
                    self.stream.close()
                except:
                    pass
            
            # Создаем новый поток для этого плеера
            self.stream = sd.OutputStream(
                samplerate=self.samplerate,
                channels=1,
                dtype='int16',
                blocksize=BUFFER_SIZE,
                latency='high'
            )
            self.stream.start()
            
            # Предварительно заполняем буфер тишиной
            pre_buffer_samples = int((PRE_BUFFER_MS / 1000.0) * self.samplerate)
            silence = np.zeros(pre_buffer_samples, dtype=np.int16)
            
            # Отправляем предварительный буфер одним куском
            self.stream.write(silence)
            
            # КРИТИЧНО: Даем время буферу заполниться и обработаться
            # Это предотвращает проглатывание начала первого слова
            time.sleep(pre_buffer_samples / self.samplerate + 0.01)  # Время буфера + 10 мс
            
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
            # Устанавливаем флаг воспроизведения
            is_playing = True
            
            # Инициализируем поток
            if not self.init_stream():
                print(f"Плеер {self.player_id}: Не удалось инициализировать аудиопоток")
                is_playing = False
                return
            
            if self.stream is None:
                print(f"Плеер {self.player_id}: Аудиопоток не создан")
                is_playing = False
                return
            
            # Получаем аудиоданные
            audio_data = self.audio_data
            total_samples = len(audio_data)
            
            if total_samples == 0:
                print(f"Плеер {self.player_id}: Нет аудиоданных для воспроизведения")
                is_playing = False
                return
            
            print(f"Плеер {self.player_id}: Начало воспроизведения, {total_samples} samples, длительность: {self.audio_duration:.2f} сек")
            
            # КРИТИЧНО: Воспроизводим всё аудио одним вызовом
            # Это гарантирует целостность данных и предотвращает прерывания
            try:
                self.stream.write(audio_data)
                
                # Ждем завершения воспроизведения с небольшим запасом
                estimated_play_time = total_samples / self.samplerate
                time.sleep(estimated_play_time + 0.05)  # + 50 мс запас
                
            except Exception as e:
                print(f"Плеер {self.player_id}: Ошибка при записи аудио: {e}")
                # Пробуем альтернативный способ: воспроизведение по чанкам
                if self.stream and self.stream.active:
                    chunk_size = CHUNK_SIZE
                    for i in range(0, total_samples, chunk_size):
                        chunk_end = min(i + chunk_size, total_samples)
                        chunk = audio_data[i:chunk_end]
                        
                        # Дополняем последний чанк
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
            # Останавливаем и закрываем поток
            if self.stream is not None:
                try:
                    # Даем время на завершение обработки
                    time.sleep(0.02)
                    self.stream.stop()
                    self.stream.close()
                except:
                    pass
                self.stream = None
            
            # Сбрасываем флаг воспроизведения
            is_playing = False

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
    return '\n'.join(lines)

def synthesize_text(text: str) -> Optional[np.ndarray]:
    """Синтез текста в аудиоданные"""
    try:
        # Предобрабатываем текст
        processed_text = preprocess_text(text)
        
        # Разбиваем на строки
        lines = processed_text.split('\n')
        
        if not lines:
            return None
        
        # Синтезируем каждую строку
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
            return None
        
        # Объединяем все строки
        audio_data = np.concatenate(all_audio_chunks)
        
        # УВЕЛИЧИВАЕМ паузы в начале и конце для предотвращения проглатывания
        # КРИТИЧНО: Увеличиваем начальную паузу
        initial_silence = int(0.1 * samplerate)  # 100 мс начальной тишины
        end_silence = int(0.05 * samplerate)     # 50 мс конечной тишины
        
        silence_start = np.zeros(initial_silence, dtype=np.int16)
        silence_end = np.zeros(end_silence, dtype=np.int16)
        
        # Также добавляем небольшой тестовый тон (1 кГц, 5 мс) в самое начало
        # Это помогает инициализировать аудиосистему
        test_tone_samples = int(0.005 * samplerate)  # 5 мс
        if test_tone_samples > 0:
            test_tone = (0.01 * np.sin(2 * np.pi * 1000 * np.linspace(0, 0.005, test_tone_samples)) * 32767).astype(np.int16)
            audio_data = np.concatenate([silence_start, test_tone, audio_data, silence_end])
        else:
            audio_data = np.concatenate([silence_start, audio_data, silence_end])
        
        print(f"Синтез завершен: {len(audio_data)} samples, начальная пауза: {initial_silence} samples ({initial_silence/samplerate*1000:.1f} мс)")
        return audio_data
        
    except Exception as e:
        print(f"Ошибка при синтезе: {e}")
        import traceback
        traceback.print_exc()
        return None

def audio_worker():
    """Фоновый рабочий поток для обработки аудио"""
    global is_playing, stop_worker
    player_counter = 0
    
    while not stop_worker:
        try:
            # Ждем текст для синтеза
            text = audio_queue.get(timeout=0.1)
            
            if text is None:
                break
            
            # Если уже воспроизводится, ждем
            if is_playing:
                print(f"Ожидание завершения текущего воспроизведения...")
                while is_playing and not stop_worker:
                    time.sleep(0.1)  # Ждем 100 мс
                time.sleep(0.2)  # Дополнительная пауза между воспроизведениями
            
            # Синтезируем аудио
            print(f"Синтез текста: '{text[:50]}...'")
            audio_data = synthesize_text(text)
            
            if audio_data is not None and len(audio_data) > 0:
                # Создаем и запускаем поток-плейер
                player_counter += 1
                player = AudioPlayerThread(audio_data, samplerate, player_counter)
                player.start()
                
                # Ждем запуска потока
                time.sleep(0.05)
            
            audio_queue.task_done()
            
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Ошибка в аудио-воркере: {e}")
            import traceback
            traceback.print_exc()

def play_audio(text: str):
    """Добавление текста в очередь на воспроизведение"""
    if text and text.strip():
        # Обрезаем слишком длинные тексты для логов
        log_text = text[:100] + "..." if len(text) > 100 else text
        print(f"Добавление в очередь: '{log_text}'")
        audio_queue.put(text)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan контекстный менеджер для управления жизненным циклом приложения.
    Заменяет устаревшие @app.on_event("startup") и @app.on_event("shutdown")
    """
    # Startup code
    print("Инициализация сервера TTS...")
    print(f"Частота дискретизации: {samplerate} Hz")
    print(f"Настройки: буфер={BUFFER_SIZE}, чанк={CHUNK_SIZE}, предбуфер={PRE_BUFFER_MS}мс")
    
    # Запускаем фоновый рабочий поток для обработки аудио
    global audio_thread
    audio_thread = threading.Thread(target=audio_worker, daemon=True)
    audio_thread.start()
    
    print("Сервер TTS запущен")
    
    yield  # Приложение работает здесь
    
    # Shutdown code
    print("Завершение работы сервера TTS...")
    
    global stop_worker
    stop_worker = True
    
    # Очищаем очередь
    while not audio_queue.empty():
        try:
            audio_queue.get_nowait()
            audio_queue.task_done()
        except:
            pass
    
    # Даем время на завершение работы
    time.sleep(1.0)
    
    print("Сервер TTS остановлен")

# Создаем FastAPI приложение с lifespan
app = FastAPI(lifespan=lifespan)

@app.post("/say")
async def say_text(request: TTSRequest, background_tasks: BackgroundTasks):
    """Обработка запроса на синтез речи"""
    background_tasks.add_task(play_audio, request.text)
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

@app.post("/test")
async def test_audio():
    """Тестовый эндпоинт для проверки воспроизведения"""
    test_text = "Привет мир! Это тестовое сообщение для проверки звука."
    background_tasks = BackgroundTasks()
    background_tasks.add_task(play_audio, test_text)
    return {"status": "test_sent", "text": test_text}

if __name__ == "__main__":
    import uvicorn
    print(f"Запуск сервера TTS на порту 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
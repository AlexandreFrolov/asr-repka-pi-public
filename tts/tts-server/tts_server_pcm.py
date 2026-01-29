import warnings
warnings.filterwarnings("ignore", message="Exception ignored from cffi callback")
import time
import numpy as np
import sounddevice as sd
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from piper import PiperVoice

# Настройки модели
MODEL_PATH = "ru_RU-irina-medium.onnx"
CONFIG_PATH = "ru_RU-irina-medium.onnx.json"

# Инициализация модели
voice = PiperVoice.load(MODEL_PATH, config_path=CONFIG_PATH)

app = FastAPI()

class TTSRequest(BaseModel):
    text: str

def play_audio(text: str):
    """Быстрый синтез без перезагрузки модели"""
    try:
        samplerate = voice.config.sample_rate
        
        # Собираем аудиоданные в список numpy массивов
        audio_chunks = []
        for audio_chunk in voice.synthesize(text):
            # Используем audio_int16_array, который содержит аудио данные в формате int16
            audio_chunks.append(audio_chunk.audio_int16_array)
        
        if audio_chunks:
            # Объединяем все чанки в один массив
            audio_data = np.concatenate(audio_chunks)
            
            print(f"Размер аудио данных: {len(audio_data)} samples")
            print(f"Частота дискретизации: {samplerate} Hz")
            print(f"Тип данных: {audio_data.dtype}")
            
            # Воспроизводим аудио через OutputStream (менее подвержено ошибкам)
            with sd.OutputStream(samplerate=samplerate, 
                               channels=1,  # моно аудио
                               dtype='int16') as stream:
                stream.write(audio_data)
            
    except Exception as e:
        print(f"Ошибка при воспроизведении: {e}")
        import traceback
        traceback.print_exc()

@app.post("/say")
async def say_text(request: TTSRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(play_audio, request.text)
    return {"status": "processing", "text": request.text}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
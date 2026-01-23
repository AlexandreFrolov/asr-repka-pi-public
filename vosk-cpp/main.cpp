#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <portaudio.h>
#include "vosk_api.h"

#define SAMPLE_RATE 16000
#define FRAMES_PER_BUFFER 2048  // Увеличенный буфер для предотвращения лагов

// Макрос для проверки ошибок PortAudio
#define CHECK_PA_ERROR(err) if(err != paNoError && err != paInputOverflowed) { \
    std::cerr << "PortAudio error: " << Pa_GetErrorText(err) << std::endl; \
    return -1; \
}

// Функция для безопасного извлечения текста из JSON строки Vosk
void print_recognized_text(const char* json_raw) {
    std::string res = json_raw;
    std::string key = "\"text\" : \"";
    size_t startPos = res.find(key);
    
    if (startPos != std::string::npos) {
        startPos += key.length();
        size_t endPos = res.find("\"", startPos);
        
        if (endPos != std::string::npos) {
            std::string text = res.substr(startPos, endPos - startPos);
            if (!text.empty()) {
                std::cout << "\r>>> РАСПОЗНАНО: " << text << std::endl;
            }
        }
    }
}

int main() {
    // 1. Инициализация Vosk
    VoskModel *model = vosk_model_new("model");
    if (!model) {
        std::cerr << "ОШИБКА: Не удалось загрузить модель из папки 'model'!" << std::endl;
        return -1;
    }
    VoskRecognizer *recognizer = vosk_recognizer_new(model, SAMPLE_RATE);

    // 2. Инициализация PortAudio
    CHECK_PA_ERROR(Pa_Initialize());

    int numDevices = Pa_GetDeviceCount();
    int inputDevice = -1;

    // Ищем USB микрофон
    for (int i = 0; i < numDevices; i++) {
        const PaDeviceInfo* deviceInfo = Pa_GetDeviceInfo(i);
        if (deviceInfo->maxInputChannels > 0 && (strstr(deviceInfo->name, "USB") || strstr(deviceInfo->name, "Audio"))) {
            std::cout << "Найдено устройство: " << deviceInfo->name << " (ID: " << i << ")" << std::endl;
            inputDevice = i;
            break;
        }
    }

    if (inputDevice == -1) {
        std::cout << "USB микрофон не найден, использую стандартный вход." << std::endl;
        inputDevice = Pa_GetDefaultInputDevice();
    }

    PaStreamParameters inputParameters;
    inputParameters.device = inputDevice;
    inputParameters.channelCount = 1;
    inputParameters.sampleFormat = paInt16;
    inputParameters.suggestedLatency = Pa_GetDeviceInfo(inputParameters.device)->defaultHighInputLatency;
    inputParameters.hostApiSpecificStreamInfo = NULL;

    PaStream *stream;
    CHECK_PA_ERROR(Pa_OpenStream(&stream, &inputParameters, NULL, SAMPLE_RATE, FRAMES_PER_BUFFER, paClipOff, NULL, NULL));
    CHECK_PA_ERROR(Pa_StartStream(stream));

    std::cout << "\n--- СИСТЕМА ГОТОВА. ГОВОРИТЕ... (Ctrl+C для выхода) ---\n" << std::endl;

    std::vector<short> buffer(FRAMES_PER_BUFFER);

    while (true) {
        // Чтение аудиоданных
        PaError err = Pa_ReadStream(stream, buffer.data(), FRAMES_PER_BUFFER);
        
        // На одноплатниках overflow — обычное дело, просто игнорируем его
        if (err != paNoError && err != paInputOverflowed) {
            std::cerr << "Критическая ошибка аудио: " << Pa_GetErrorText(err) << std::endl;
            break;
        }

        // Передача данных в нейросеть
        int is_final = vosk_recognizer_accept_waveform_s(recognizer, buffer.data(), FRAMES_PER_BUFFER);
        
        if (is_final) {
            print_recognized_text(vosk_recognizer_result(recognizer));
        } else {
            // Опционально: можно выводить промежуточные результаты
            // std::cout << " Слушаю..." << "\r" << std::flush;
        }
    }

    // 3. Очистка
    Pa_StopStream(stream);
    Pa_CloseStream(stream);
    Pa_Terminate();
    vosk_recognizer_free(recognizer);
    vosk_model_free(model);

    return 0;
}

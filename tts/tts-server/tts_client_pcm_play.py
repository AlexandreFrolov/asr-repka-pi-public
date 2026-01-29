import sys
import requests
import json

def main():
    # Проверяем наличие аргумента командной строки
    if len(sys.argv) != 2:
        print("Использование: python client.py <путь_к_файлу>")
        print("Пример: python client.py text.txt")
        sys.exit(1)
    
    # Получаем путь к файлу из аргументов
    file_path = sys.argv[1]
    
    try:
        # Читаем текст из файла
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read().strip()
        
        # Проверяем, что файл не пустой
        if not text:
            print("Файл пуст!")
            sys.exit(1)
        
        # Формируем данные для отправки
        data = {"text": text}
        
        # URL сервера (замените на актуальный, если отличается)
        url = "http://192.168.0.18:8000/say"
        
        # Отправляем POST запрос
        response = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            data=json.dumps(data, ensure_ascii=False).encode('utf-8')
        )
        
        # Проверяем ответ
        if response.status_code == 200:
            print("Запрос успешно отправлен!")
            print(f"Ответ сервера: {response.text}")
        else:
            print(f"Ошибка: {response.status_code}")
            print(f"Ответ: {response.text}")
    
    except FileNotFoundError:
        print(f"Файл не найден: {file_path}")
        sys.exit(1)
    except requests.exceptions.ConnectionError:
        print("Не удалось подключиться к серверу. Проверьте адрес и порт.")
        sys.exit(1)
    except Exception as e:
        print(f"Произошла ошибка: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
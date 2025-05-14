from ultralytics import YOLO
from collections import Counter
from pathlib import Path
import os
import shutil

test_dir = 'image_testing'
output_dir = 'output_dir'
supported_extensions = {'.jpg', ".jpeg", ".png"}

def setup_dir():
    """"Проверим существование каталога для исходных изображений"""
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)
        print(f'[+] Директория {test_dir} была создана')
    else:
        print(f'[INFO] Директория {test_dir} уже существует, создание директории пропускается.')

    """Проверим существование директории для размеченых изображений"""
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        print(f'[+] Директория {output_dir} была создана')
    else:
        print(f'[INFO] Директория {output_dir} уже существует, создание директории пропускается.')

def image_path(directory: str) -> list:
    """Переносим изображения вне каталога в test_dir"""
    
    image_search = Path(directory)
    dest_path = Path(test_dir) / image_search.name

    try: 
        if not image_search.exists():
            raise FileNotFoundError(f"Исходный файл {image_search} не найден")
          
        if image_search.parent != Path(test_dir):
            shutil.copy2(image_search, dest_path)
            os.remove(image_search)
            print(f"[+] Изображение перемещено в {test_dir}")

        return str(dest_path)
        
    except Exception as e:
        print(f"[!] Ошибка при обработке изображения: {str(e)}")
        raise

def process_images(model: YOLO):
    """Обрабатываем изображения в каталоге test_dir"""
    test_path = Path(test_dir)

    # Получаем список изображений
    image_files = [
        
        j for j in test_path.iterdir()
        if j.is_file() and j.suffix.lower() in supported_extensions
    ]

    if not image_files:
        print(f'[INFO] В каталоге {test_dir} нет изображений для обработки')
        return
    print(f'[INFO] Найдено {len(image_files)} изображений для обработки')

    # Проходимся по найденым файлам
    for image_path in image_files:
        try:
            print(f'\n[Initialization] Начало обработки: {image_path.name}')
            DetectObjects(model, str(image_path))
            print(f"[+] {image_path.name} успешно обработано")

        except Exception as e:
            print(f"[INFO] Ошибка при обработке {image_path.name}: {str(e)}")

def DetectObjects(model: YOLO, image_path: str) -> None:
    """
    param model: загруженная модель YOLO
    param image_path: путь к изображению, на котором нужно произвести детекцию.
    """
    try:
        results = model(image_path, verbose=False)[0]

        if results.names and results.boxes is not None:
            # Получаем список идентификаторов классов объектов
            labels = results.boxes.cls.tolist()
            # Преобразуем ID в названии классов
            label_names = [results.names[int(cls)] for cls in labels]
            # Считаем количество каждого уникального объекта
            counts = Counter(label_names)
          
            print("[INFO] Обнаруженные объекты:")

            for label, count in counts.items():
                print(f'[+] {label}: {count}')
        else:
            print("[!] Объекты не обнаружены!")
        
        # Путь для сохранения
        save_filename = f'RESULT_{Path(image_path).stem}.png'
        save_path = Path(output_dir) /save_filename

        results.save(filename=str(save_path))
        print(f"[+] Результат сохранен в: {save_path}")

    except Exception as e:
        print(f'[INFO] Ошибка при детекции объектов: {str(e)}')
        raise

def main():
    setup_dir()
    model = YOLO('yolov8n.pt')

    # Обработка всех изображений в директории
    for img in Path('.').glob('*'):
        if img.suffix.lower() in supported_extensions:
            image_path(str(img))
    
    process_images(model)
    
if __name__=="__main__":
    main()
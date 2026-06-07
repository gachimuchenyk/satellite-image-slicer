import os
import json
import cv2
import numpy as np

# Шляхи до твоїх папок
INPUT_IMG_DIR = "yuma_4k_images"
INPUT_JSON_DIR = "yuma_jsons"
OUTPUT_DIR = "yuma_dataset_tiled"
TILE_SIZE = 640
OVERLAP = 64  # Невеликий напуск, щоб не "різати" машини навпіл

os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "labels"), exist_ok=True)


def tile_dataset():
    img_files = [f for f in os.listdir(INPUT_IMG_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))]

    for img_name in img_files:
        img_path = os.path.join(INPUT_IMG_DIR, img_name)
        json_path = os.path.join(INPUT_JSON_DIR, os.path.splitext(img_name)[0] + ".json")

        if not os.path.exists(json_path):
            print(f"Пропуск {img_name}: немає файлу розмітки")
            continue

        img = cv2.imread(img_path)
        h, w, _ = img.shape

        with open(json_path, 'r') as f:
            data = json.load(f)

        # Проходимо по картинці сіткою
        for y in range(0, h - TILE_SIZE + 1, TILE_SIZE - OVERLAP):
            for x in range(0, w - TILE_SIZE + 1, TILE_SIZE - OVERLAP):
                tile_name = f"{os.path.splitext(img_name)[0]}_y{y}_x{x}"

                # Вирізаємо тайл
                tile = img[y:y + TILE_SIZE, x:x + TILE_SIZE]

                # Фільтруємо бокси, що потрапляють у цей тайл
                new_labels = []
                for shape in data['shapes']:
                    points = np.array(shape['points'])
                    # Координати боксу
                    x1, y1 = np.min(points, axis=0)
                    x2, y2 = np.max(points, axis=0)

                    # Перевіряємо, чи центр машини в межах тайла
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

                    if x <= cx < x + TILE_SIZE and y <= cy < y + TILE_SIZE:
                        # Перераховуємо в координати тайла і нормалізуємо для YOLO
                        nx = (cx - x) / TILE_SIZE
                        ny = (cy - y) / TILE_SIZE
                        nw = (x2 - x1) / TILE_SIZE
                        nh = (y2 - y1) / TILE_SIZE

                        # YOLO format: class_id x_center y_center width height
                        new_labels.append(f"0 {nx:.6f} {ny:.6f} {nw:.6f} {nh:.6f}")

                # Зберігаємо тільки якщо є машини або як фоновий тайл (опціонально)
                if new_labels:
                    cv2.imwrite(os.path.join(OUTPUT_DIR, "images", tile_name + ".jpg"), tile)
                    with open(os.path.join(OUTPUT_DIR, "labels", tile_name + ".txt"), "w") as f:
                        f.write("\n".join(new_labels))

    print("Готово! Твій нарізаний датасет чекає в папці yuma_dataset_tiled")


if __name__ == "__main__":
    tile_dataset()
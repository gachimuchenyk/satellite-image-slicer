#!/usr/bin/env python3
"""
slice_satellite.py — нарізка 4K супутникових знімків для тренування YOLO
Підтримує лейбли типу: white_sedan, grey_suv, old_musclecar і т.п.
"""

import os
import json
import cv2
import numpy as np
import random

# ── Налаштування ────────────────────────────────────────────────────────────
INPUT_IMG_DIR   = r"E:\2magisterska\DataSet_Yuma\Yuma4sectorsScreenshots\YumaА\YumaA4"
INPUT_JSON_DIR  = r"E:\2magisterska\DataSet_Yuma\Yuma4sectorsScreenshots\YumaА\YumaA4"
OUTPUT_DIR      = r"E:\2magisterska\DataSet_Yuma\Yuma4sectorsScreenshots\YumaА\YumaA4Sliced"
TILE_SIZE       = 640
OVERLAP         = 64
SAVE_EMPTY_PROB = 0.2   # 20% порожніх тайлів зберігаємо
MIN_BOX_SIZE    = 0.01  # мінімальний розмір bbox (нормалізований)
JPEG_QUALITY    = 95
# ────────────────────────────────────────────────────────────────────────────

os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "labels"), exist_ok=True)

# Класи для data.yaml
# Класи для data.yaml
CLASS_NAMES = {
    0: "heavy_vehicle",          # звичайні фури та вантажівки
    1: "pickup",                 # пікапи
    2: "suv_jeep",               # джипи та кросовери
    3: "van",                    # фургони
    4: "hatchback",              # хетчбеки
    5: "sedan_car",              # седани та стандартні авто
    6: "trailer",                # причепи
    7: "bus",                    # автобуси
    8: "musclecar",              # маслкари
    9: "rv_camper",              # будинки на колесах
    10: "junk_car",              # покинуті/зруйновані авто (junk)
    11: "construction_viechle",  # будівельна техніка (bucket, construction)
    12: "sixbysix",              # унікальні 6x6
    13: "motorcycle"
}


def parse_label(raw: str) -> int:
    """
    Деталізований парсер. ПОРЯДОК ПЕРЕВІРКИ МАЄ ЗНАЧЕННЯ!
    """
    raw = raw.lower().strip()
    # 13. Мотоцикли
    if "motorcycle" in raw:
        return 13
    # 12. Унікальна техніка 6x6
    if "sixbysix" in raw:
        return 12
    # 11. Будівельна та спецтехніка (ПЕРЕД вантажівками, бо є слово truck)
    if any(x in raw for x in ["construction_viechle", "construction_vehicle", "bucket"]):
        return 11
    # 10. Автобрухт / Покинуті
    if "junk" in raw:
        return 10
    # 0. Важка техніка (звичайні вантажівки та фури)
    if any(x in raw for x in ["semitruck", "semi_truck", "truck"]):
        return 0
    # 9. Будинки на колесах (ПЕРЕД венами)
    if any(x in raw for x in ["camper", "house_on_wheels", "rv"]):
        return 9
    # 3. Звичайні вени
    if any(x in raw for x in ["van", "ambulance"]):
        return 3
    # 8. Маслкари (ПЕРЕД перевіркою на car)
    if any(x in raw for x in ["musclecar", "muclecar", "muscleca"]):
        return 8
    # 1. Пікапи
    if any(x in raw for x in ["pickup", "blackp", "blue_p"]):
        return 1
    # 2. Джипи
    if any(x in raw for x in ["suv", "jeep"]):
        return 2
    # 4. Хетчбеки
    if any(x in raw for x in ["hatchback", "mini"]):
        return 4
    # 6. Причепи
    if "trailer" in raw:
        return 6
    # 7. Автобуси
    if "bus" in raw:
        return 7
    # 5. Седани та базові авто (В САМОМУ КІНЦІ, бо слово car є в багатьох інших словах)
    if any(x in raw for x in ["sedan", "car", "sedn", "whtie_s", "classic", "lowrider"]):
        return 5

    return None


def clip_box(x1, y1, x2, y2, tx, ty, tile_size):
    """Обрізає bbox до меж тайлу. Повертає None якщо bbox поза тайлом."""
    x1c = max(x1, tx)
    y1c = max(y1, ty)
    x2c = min(x2, tx + tile_size)
    y2c = min(y2, ty + tile_size)
    if x2c <= x1c or y2c <= y1c:
        return None
    return x1c, y1c, x2c, y2c


def make_tiles(w, h, tile_size, overlap):
    """
    Генерує координати тайлів з покриттям ВСЬОГО знімка включно з краями.
    Без цього машини біля правого/нижнього краю втрачаються.
    """
    step = tile_size - overlap
    xs = list(range(0, w - tile_size, step))
    ys = list(range(0, h - tile_size, step))
    # Явно додаємо правий і нижній край
    if not xs or xs[-1] + tile_size < w:
        xs.append(w - tile_size)
    if not ys or ys[-1] + tile_size < h:
        ys.append(h - tile_size)
    return xs, ys


def save_yaml():
    """Зберігає data.yaml для YOLO."""
    yaml_path = os.path.join(OUTPUT_DIR, "data.yaml")
    lines = [
        f"path: {os.path.abspath(OUTPUT_DIR)}",
        "train: images",
        "val: images",   # заміни на свій val split якщо є
        "",
        f"nc: {len(CLASS_NAMES)}",
        "names:",
    ]
    for i, name in CLASS_NAMES.items():
        lines.append(f"  {i}: {name}")
    with open(yaml_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Збережено: {yaml_path}")


def tile_dataset():
    img_files = [
        f for f in os.listdir(INPUT_IMG_DIR)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]

    if not img_files:
        print(f"[!] Не знайдено зображень у папці: {INPUT_IMG_DIR}")
        return

    tile_counter   = 0
    label_counter  = 0
    skipped_imgs   = 0
    unknown_labels = set()

    for img_name in img_files:
        img_path  = os.path.join(INPUT_IMG_DIR, img_name)
        json_path = os.path.join(INPUT_JSON_DIR,
                                 os.path.splitext(img_name)[0] + ".json")

        if not os.path.exists(json_path):
            print(f"  [пропуск] {img_name}: немає JSON")
            skipped_imgs += 1
            continue

        # Використовуємо numpy для читання файлу, щоб ігнорувати проблеми з кирилицею у Windows
        img_array = np.fromfile(img_path, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            print(f"  [пропуск] {img_name}: не вдалося прочитати")
            skipped_imgs += 1
            continue

        h, w = img.shape[:2]

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        stem = os.path.splitext(img_name)[0]
        xs, ys = make_tiles(w, h, TILE_SIZE, OVERLAP)

        print(f"  {img_name} ({w}x{h}) -> {len(xs)}x{len(ys)} = {len(xs)*len(ys)} тайлів")

        for y in ys:
            for x in xs:
                tile = img[y:y + TILE_SIZE, x:x + TILE_SIZE]
                labels = []

                for shape in data.get('shapes', []):
                    raw_label = shape.get('label', '').strip()
                    if not raw_label:
                        continue

                    class_id = parse_label(raw_label)
                    if class_id is None:
                        unknown_labels.add(raw_label)
                        continue  # Пропускаємо запис цього об'єкта

                    # Логуємо невідомі лейбли (для дебагу)
                    class_id = parse_label(raw_label)
                    if class_id is None:
                        unknown_labels.add(raw_label)
                        continue  # Пропускаємо запис цього об'єкта

                    points = np.array(shape['points'], dtype=np.float32)
                    x1, y1b = np.min(points, axis=0)
                    x2, y2b = np.max(points, axis=0)

                    clipped = clip_box(x1, y1b, x2, y2b, x, y, TILE_SIZE)
                    if clipped is None:
                        continue

                    x1c, y1c, x2c, y2c = clipped

                    cx = ((x1c + x2c) / 2 - x) / TILE_SIZE
                    cy = ((y1c + y2c) / 2 - y) / TILE_SIZE
                    bw = (x2c - x1c)            / TILE_SIZE
                    bh = (y2c - y1c)            / TILE_SIZE

                    if bw < MIN_BOX_SIZE or bh < MIN_BOX_SIZE:
                        continue

                    labels.append(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                    label_counter += 1

                if labels or random.random() < SAVE_EMPTY_PROB:
                    tile_name = f"{stem}_x{x}_y{y}"
                    # Бронебійне збереження через numpy (ігнорує кирилицю)
                    save_path = os.path.join(OUTPUT_DIR, "images", tile_name + ".jpg")
                    is_success, buffer = cv2.imencode(".jpg", tile, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                    if is_success:
                        buffer.tofile(save_path)
                    with open(os.path.join(OUTPUT_DIR, "labels",
                                           tile_name + ".txt"), "w") as f:
                        if labels:
                            f.write("\n".join(labels) + "\n")
                    tile_counter += 1

    print(f"""
Готово!
  Тайлів збережено:   {tile_counter}
  Лейблів всього:     {label_counter}
  Знімків пропущено:  {skipped_imgs}
""")

    if unknown_labels:
        print(f"  [!] Невідомі лейбли (перевір вручну): {sorted(unknown_labels)}")

    save_yaml()



if __name__ == "__main__":
    tile_dataset()

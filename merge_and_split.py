import os
import shutil
import random

# ── Налаштування ────────────────────────────────────────────────────────────
# Вкажи шляхи до твоїх трьох нарізаних папок:
SOURCE_DIRS = [
    r"E:\2magisterska\DataSet_Yuma\Yuma4sectorsScreenshots\YumaА\YumaA1Sliced",
    r"E:\2magisterska\DataSet_Yuma\Yuma4sectorsScreenshots\YumaА\YumaA2Sliced",
    r"E:\2magisterska\DataSet_Yuma\Yuma4sectorsScreenshots\YumaА\YumaA3Sliced"
]

# Куди зберегти фінальний об'єднаний датасет для тренування:
OUTPUT_DIR = r"E:\2magisterska\DataSet_Yuma\Yuma_Master_Dataset"

# Пропорція розбиття (85% на навчання, 15% на валідацію)
SPLIT_RATIO = 0.85
# ────────────────────────────────────────────────────────────────────────────

CLASS_NAMES = {
    0: "heavy_vehicle", 1: "pickup", 2: "suv_jeep", 3: "van",
    4: "hatchback", 5: "sedan_car", 6: "trailer", 7: "bus",
    8: "musclecar", 9: "rv_camper", 10: "junk_car",
    11: "construction_machine", 12: "sixbysix", 13: "motorcycle"
}

def create_dirs():
    # Створюємо структуру папок для YOLO
    for split in ['train', 'val']:
        os.makedirs(os.path.join(OUTPUT_DIR, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, 'labels', split), exist_ok=True)

def merge_and_split():
    print("=== Об'єднання та розбиття датасету ===")
    create_dirs()

    all_pairs = []

    # Збираємо всі файли з трьох папок
    for i, source_dir in enumerate(SOURCE_DIRS):
        img_dir = os.path.join(source_dir, "images")
        lbl_dir = os.path.join(source_dir, "labels")

        if not os.path.exists(img_dir) or not os.path.exists(lbl_dir):
            print(f"[!] Папку не знайдено: {source_dir}, пропускаємо...")
            continue

        images = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
        print(f"Знайдено {len(images)} тайлів у папці {i+1}...")

        # Додаємо префікс до імені файлу, щоб імена з різних папок не перекривали одне одного
        prefix = f"sec{i+1}_"

        for img_name in images:
            txt_name = img_name.replace('.jpg', '.txt')
            img_path = os.path.join(img_dir, img_name)
            txt_path = os.path.join(lbl_dir, txt_name)

            if os.path.exists(txt_path):
                # Зберігаємо шляхи і нове ім'я
                all_pairs.append((img_path, txt_path, prefix + img_name, prefix + txt_name))

    if not all_pairs:
        print("Помилка: не знайдено жодного файлу для об'єднання!")
        return

    # Рандомно перемішуємо весь список
    print("\nПеремішуємо дані...")
    random.seed(42)  # Фіксуємо seed для повторюваності
    random.shuffle(all_pairs)

    # Розраховуємо кількість
    train_count = int(len(all_pairs) * SPLIT_RATIO)
    train_pairs = all_pairs[:train_count]
    val_pairs = all_pairs[train_count:]

    print(f"Всього тайлів: {len(all_pairs)}")
    print(f"Йде в Train: {len(train_pairs)}")
    print(f"Йде в Val:   {len(val_pairs)}\n")

    # Функція для копіювання
    def copy_data(pairs, split_name):
        print(f"Копіюємо файли у {split_name.upper()}...")
        for img_src, txt_src, img_dst_name, txt_dst_name in pairs:
            shutil.copy(img_src, os.path.join(OUTPUT_DIR, 'images', split_name, img_dst_name))
            shutil.copy(txt_src, os.path.join(OUTPUT_DIR, 'labels', split_name, txt_dst_name))

    copy_data(train_pairs, 'train')
    copy_data(val_pairs, 'val')

    # Створюємо фінальний data.yaml
    yaml_path = os.path.join(OUTPUT_DIR, "data.yaml")
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(f"path: {OUTPUT_DIR}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n\n")
        f.write(f"nc: {len(CLASS_NAMES)}\n")
        f.write("names:\n")
        for idx in range(len(CLASS_NAMES)):
            f.write(f"  {idx}: {CLASS_NAMES[idx]}\n")

    print(f"\n✅ ГОТОВО! Мега-датасет створено: {OUTPUT_DIR}")
    print(f"Файл конфігурації: {yaml_path}")

if __name__ == "__main__":
    merge_and_split()
import os
import cv2
import numpy as np

# ── Налаштування ────────────────────────────────────────────────────────────
# Вкажи шлях до папки з нарізаним датасетом
DATASET_DIR = r"E:\2magisterska\DataSet_Yuma\Yuma4sectorsScreenshots\YumaА\YumaA3Sliced"
# ────────────────────────────────────────────────────────────────────────────

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
    11: "construction_machine",  # будівельна техніка
    12: "sixbysix",               # унікальні 6x6
    13: "motorcycle"
}


def check_dataset():
    images_dir = os.path.join(DATASET_DIR, "images")
    labels_dir = os.path.join(DATASET_DIR, "labels")

    image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]

    if not image_files:
        print("Картинок не знайдено!")
        return

    print("=== Режим перевірки датасету ===")
    print("УВАГА: Перемкни клавіатуру на АНГЛІЙСЬКУ (EN)!")
    print("  [ d ] - наступна картинка")
    print("  [ a ] - попередня картинка")
    print("  [ e ] - стрибок на 100 картинок ВПЕРЕД")
    print("  [ w ] - стрибок на 100 картинок НАЗАД")
    print("  [ q ] або [ Esc ] - вийти\n")

    # Створюємо вікно один раз
    win_name = "Dataset Checker"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    i = 0
    while i < len(image_files):
        img_name = image_files[i]
        txt_name = img_name.replace('.jpg', '.txt')

        img_path = os.path.join(images_dir, img_name)
        txt_path = os.path.join(labels_dir, txt_name)

        # Читаємо картинку бронебійним способом
        img_array = np.fromfile(img_path, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        h, w = img.shape[:2]

        # Малюємо рамки, якщо є файл розмітки
        if os.path.exists(txt_path):
            with open(txt_path, 'r') as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id = int(parts[0])
                    cx, cy, bw, bh = map(float, parts[1:])

                    x_center, y_center = int(cx * w), int(cy * h)
                    box_w, box_h = int(bw * w), int(bh * h)

                    x1 = int(x_center - box_w / 2)
                    y1 = int(y_center - box_h / 2)
                    x2 = int(x_center + box_w / 2)
                    y2 = int(y_center + box_h / 2)

                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    label = CLASS_NAMES.get(class_id, "unknown")
                    cv2.putText(img, label, (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # Динамічно міняємо назву вікна
        title = f"File: {img_name} | Image {i + 1} of {len(image_files)}"
        cv2.setWindowTitle(win_name, title)

        cv2.imshow(win_name, img)

        # Обробка клавіш
        key = cv2.waitKey(0) & 0xFF

        if key in [ord('q'), 27]:  # 'q' або Esc
            break
        elif key == ord('a'):  # 'a' - назад на 1
            i = max(0, i - 1)
        elif key == ord('d'):  # 'd' - вперед на 1
            if i < len(image_files) - 1:
                i += 1
            else:
                print("Це остання картинка!")
        elif key == ord('e'):  # 'e' - стрибок на 100 вперед
            i = min(len(image_files) - 1, i + 100)
        elif key == ord('w'):  # 'w' - стрибок на 100 назад
            i = max(0, i - 100)
        # Будь-які інші клавіші ігноруються

    cv2.destroyAllWindows()


if __name__ == "__main__":
    check_dataset()
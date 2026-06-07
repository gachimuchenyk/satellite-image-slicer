from ultralytics import YOLO

# 1. Завантажуємо ТВІЙ натренований "мозок"
# Встав сюди правильний шлях до твого файлу best.pt
model = YOLO(r"E:\2magisterska\yuma_model\weights\best.pt")

# 2. Даємо йому нову картинку на перевірку
# Встав сюди шлях до картинки, яку хочеш перевірити
image_path = r"E:\2magisterska\DataSet_Yuma\Yuma4sectorsScreenshots\YumaА\YumaA4Sliced\images\1_x0_y0.jpg"

print("Шукаю техніку...")

# 3. Запускаємо сканування
results = model.predict(
    source=image_path,
    conf=0.38,      # Поріг впевненості. 0.38 означає: "показуй рамку, якщо впевнений хоча б на 38%"
    iou=0.45,       # Допомагає не малювати дві рамки на одній машині
    save=True,      # ЗБЕРЕГТИ нову картинку з намальованими рамками
    show=False,      # Не відкривати вікно (бо через AnyDesk може глючити), ми просто подивимось збережений файл
    save_txt=True
)

print("✅ Магія завершена! Результат збережено у папку: runs/detect/predict")
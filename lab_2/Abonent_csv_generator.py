import csv
import random
from faker import Faker

# Создайте экземпляр Faker для генерации случайных имен и номеров телефонов
fake = Faker()

# Определите поля, которые будут в вашем CSV файле
fields = ["Age", "Consumption", "Call_duration", "Day_calls", "Evening_calls", "Night_calls", "Intercity_calls",
          "International_calls", "Internet", "Sms"]

# Генерируйте случайные данные и записывайте их в CSV файл
with open("Абоненты.csv", "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fields)
    writer.writeheader()

    for _ in range(50):  # Генерируем данные для 100 абонентов (измените по необходимости)
        writer.writerow({
            "Age": random.randint(18, 80),
            "Consumption": round(random.uniform(0, 2000.0), 2),
            "Call_duration": random.randint(0, 200),
            "Day_calls": random.randint(0, 500),
            "Evening_calls": random.randint(0, 500),
            "Night_calls": random.randint(0, 500),
            "Intercity_calls": random.randint(0, 500),
            "International_calls": random.randint(0, 500),
            "Internet": round(random.uniform(0.0, 100.0), 2),
            "Sms": random.randint(0, 300)
        })

print("CSV файл успешно создан.")
import shutil

import keras
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from keras.src.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.datasets import fashion_mnist
from keras.preprocessing.image import ImageDataGenerator

matplotlib.use('TkAgg')

y_data = pd.read_csv("war_TCHBYGON/war_tech_gont-export.csv")

# Путь к вашей общей папке с изображениями
input_folder = 'war_TCHBYGON/obshaya_papk'

# Путь к вашей общей папке с изображениями
main_folder = 'war_TCHBYGON'

# Путь к обучающей папке
train_folder = os.path.join(main_folder, 'train')

# Путь к тестовой папке
test_folder = os.path.join(main_folder, 'test')

# Проверка существования папок train и test
if not os.path.exists(train_folder) or not os.path.exists(test_folder):
    # Если одна из папок отсутствует, выполняем разделение данных и копирование
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    # Проход по подпапкам (классам) в общей папке
    for class_folder in os.listdir(input_folder):
        class_path = os.path.join(input_folder, class_folder)
        if os.path.isdir(class_path):
            # Создание подпапок для обучающей и тестовой выборок
            train_class_path = os.path.join(train_folder, class_folder)
            os.makedirs(train_class_path, exist_ok=True)

            test_class_path = os.path.join(test_folder, class_folder)
            os.makedirs(test_class_path, exist_ok=True)

            # Получение списка файлов в текущей подпапке (классе)
            all_images = os.listdir(class_path)

            # Разделение файлов на обучающую и тестовую выборки для текущего класса
            train_images, test_images = train_test_split(all_images, test_size=0.2, random_state=42)

            # Копирование изображений в соответствующие папки
            for image in train_images:
                shutil.copy(os.path.join(class_path, image), os.path.join(train_class_path, image))

            for image in test_images:
                shutil.copy(os.path.join(class_path, image), os.path.join(test_class_path, image))
else:
    print("Папки 'train' и 'test' уже существуют.")







train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.15
)

train_generator = train_datagen.flow_from_directory(
    train_folder,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

test_generator = train_datagen.flow_from_directory(
    test_folder,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Генератор данных для валидации
validation_generator = train_datagen.flow_from_directory(
    train_folder,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'  # указание, что это генератор данных для валидации
)

# Получение пар изображений и меток из обучающего генератора
images, numeric_labels = train_generator.next()

# Получение словаря сопоставления числовых меток классов и их названий
class_labels = {v: k for k, v in train_generator.class_indices.items()}

# Преобразование числовых меток в названия классов
class_names = [class_labels[label] for label in numeric_labels.argmax(axis=1)]

fig = plt.figure(figsize=(10, 5))
plt.subplots_adjust(hspace=0.5)
fig.suptitle('20 первых элеметов выборки')
for i in range(20):
    plt.subplot(4, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.title(f'Class: {class_names[i]}')
    plt.imshow(images[i])


# Формируем модель НС.
model = Sequential([
    Conv2D(128, (5, 5), input_shape=(224, 224, 3), activation='relu'),
    MaxPooling2D((3, 3), strides=2),
    Conv2D(128, (3, 3), padding='same', activation='relu'),
    MaxPooling2D((2, 2), strides=2),
    Conv2D(128, (3, 3), padding='same', activation='relu'),
    MaxPooling2D((2, 2), strides=2),
    Dropout(0.4),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.4),
    Dense(8, activation='softmax')])


# Компилируем модель
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)

# Обучаем НС
log = model.fit(train_generator, epochs=10, validation_data=validation_generator)

# Оцениваем качество модели
err = model.evaluate(test_generator)
print('Значение ошибки: ', err)

# Выводим график ошибки
fig2 = plt.figure(figsize=(5, 5))
fig2.suptitle('График ошибки')
plt.plot(log.history['loss'], label='Потери')
plt.plot(log.history['val_loss'], label='Валидация потерь')
plt.legend()
plt.grid(True)

# Отображаем на графике точность
fig3 = plt.figure(figsize=(7, 5))
fig3.suptitle('График точности')
plt.plot(log.history['accuracy'], label='Точность')
plt.plot(log.history['val_accuracy'], label='Валидация точности')
plt.legend()
plt.grid(True)




plt.show()

# # Формируем исходные данные, импортируем их из mnist датасета
# (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
#
# # Название классов для классификации
# classes = ['футболка', 'брюки', 'свитер', 'платье', 'пальто', 'туфли', 'рубашка',
#            'кроссовки', 'сумка', 'ботинки']
#
# # Выводим исходные данные
# fig = plt.figure(figsize=(10, 5))
# plt.subplots_adjust(hspace=0.5)
# fig.suptitle('25 первых элеметов выборки')
# for i in range(25):
#     plt.subplot(5, 5, i + 1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.title(f'{classes[y_train[i]]}')
#     plt.imshow(x_train[i], cmap=plt.cm.binary)

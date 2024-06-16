import keras
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.datasets import fashion_mnist

matplotlib.use('TkAgg')

# Формируем исходные данные, импортируем их из mnist датасета
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Название классов для классификации
classes = ['футболка', 'брюки', 'свитер', 'платье', 'пальто', 'туфли', 'рубашка',
           'кроссовки', 'сумка', 'ботинки']

# Выводим исходные данные
fig = plt.figure(figsize=(10, 5))
plt.subplots_adjust(hspace=0.5)
fig.suptitle('25 первых элеметов выборки')
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.title(f'{classes[y_train[i]]}')
    plt.imshow(x_train[i], cmap=plt.cm.binary)

# Стандартизируем входные значения вектора x
x_train = x_train / 255
x_test = x_test / 255

# Преобразуем выходные значения вектора y в категориальную форму
y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

# Добавляем третью ось к чёрно-белым исходным изображениям
x_train = np.expand_dims(x_train, axis=3)
x_test = np.expand_dims(x_test, axis=3)

# Формируем модель НС.
model = Sequential([
    Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2), strides=2),
    Conv2D(64, (3, 3), padding='same', activation='relu'),
    MaxPooling2D((2, 2), strides=2),
    Conv2D(64, (3, 3), padding='same', activation='relu'),
    MaxPooling2D((2, 2), strides=2),
    Dropout(0.4),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.4),
    Dense(10, activation='softmax')])


# Компилируем модель
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Обучаем НС
log = model.fit(x_train, y_train_cat, batch_size=64, epochs=15, validation_split=0.2)

# Оцениваем качество модели
err = model.evaluate(x_test, y_test_cat)
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

# Проверям правильность распознования изображений
fig4 = plt.figure(figsize=(8, 8))
plt.subplots_adjust(hspace=0.5)
fig4.suptitle('Проверка на тестовых данных')
for n in range(10):
    x = np.expand_dims(x_test[n], axis=0)
    res = model.predict(x)
    plt.subplot(5, 2, n + 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_test[n], cmap=plt.cm.binary)
    plt.title(f'Объект: {classes[y_test[n]]}\nРезультат классификации: {classes[np.argmax(res)]}')

# Проверям правильность распознования изображений без вывода изображений
pred_2 = model.predict(x_test)
pred_2 = np.argmax(pred_2, axis=1)
print('Результат классификации')
print(pred_2[:30])
print(y_test[:30])


plt.show()

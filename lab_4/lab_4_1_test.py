import keras
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.utils import to_categorical
from keras.datasets import mnist

matplotlib.use('TkAgg')


def test_2(epoch: int, data_size: int, test_size_percent: float):
    # 2.Множественная регрессия
    # Формируем исходные данные
    x_data_1 = np.random.randint(10, size=(data_size, 2))  # Генерируем случайные точки
    y_data_1 = x_data_1[:, 0] ** 2 + x_data_1[:, 0] * x_data_1[:, 1]  # Вычисляем y

    # Разделение данных на обучающую и тестовую выборку
    x_train_1, x_test_1, y_train_1, y_test_1 = train_test_split(x_data_1, y_data_1, test_size=test_size_percent)

    # Построить последовательную модель
    model_1 = Sequential()

    # Добавляем к модели слои
    model_1.add(Dense(units=20, input_dim=2, activation='relu'))
    model_1.add(Dense(units=1, activation='linear'))

    # Компилируем модель
    model_1.compile(optimizer='adam', loss='mse')

    # Обучаем сеть
    log_1 = model_1.fit(x_train_1, y_train_1, epochs=epoch, batch_size=20, verbose=0)

    # Оцениваем качество модели
    err_1 = model_1.evaluate(x_test_1, y_test_1)
    print('Ошибка: ', err_1)

    # Отображаем на графике ошибку
    fig3 = plt.figure(figsize=(5, 5))
    fig3.suptitle('График ошибки')
    plt.plot(log_1.history['loss'], label='Потери')
    plt.legend()
    plt.grid(True)

    # Проводим предсказание
    x1 = 2
    x2 = 4
    print(f'Вход: x1{x1} x2{x2}')
    pred_1 = model_1.predict([[x1, x2]])
    print('Значение функции на основе нейронной сети: ', pred_1)
    print('Реальный результат: ', x1 ** 2 + x1 * x2)


def test_2_1(epoch: int, data_size: int, test_size_percent: float):
    # 2.Множественная регрессия
    # Формируем исходные данные
    x_data_1 = np.random.randint(10, size=(data_size, 2))  # Генерируем случайные точки
    y_data_1 = x_data_1[:, 0] ** 2 + x_data_1[:, 0] * x_data_1[:, 1]  # Вычисляем y

    # Разделение данных на обучающую и тестовую выборку
    x_train_1, x_test_1, y_train_1, y_test_1 = train_test_split(x_data_1, y_data_1, test_size=test_size_percent)

    # Построить последовательную модель
    model_1 = Sequential()

    # Добавляем к модели слои
    model_1.add(Dense(units=20, input_dim=2, activation='relu'))
    model_1.add(Dense(units=30, input_dim=2, activation='relu'))
    model_1.add(Dense(units=1, activation='linear'))

    # Компилируем модель
    model_1.compile(optimizer='adam', loss='mse')

    # Обучаем сеть
    log_1 = model_1.fit(x_train_1, y_train_1, epochs=epoch, batch_size=20, verbose=0)

    # Оцениваем качество модели
    err_1 = model_1.evaluate(x_test_1, y_test_1)
    print('Ошибка: ', err_1)

    # Отображаем на графике ошибку
    fig = plt.figure(figsize=(5, 5))
    fig.suptitle('График ошибки')
    plt.plot(log_1.history['loss'], label='Потери')
    plt.legend()
    plt.grid(True)

    # Проводим предсказание
    x1 = 2
    x2 = 4
    print(f'Вход: x1: {x1} x2: {x2}')
    pred_1 = model_1.predict([[x1, x2]])
    print('Значение функции на основе нейронной сети: ', pred_1)
    print('Реальный результат: ', x1 ** 2 + x1 * x2)


def test_3(epoch: int, data_size: int, test_size_percent: float):
    # 3.Бинарная классификация. Классификация чётных и нечётных цифр
    # Генерируем случайные данные от 0 до 9
    x_data_2 = np.random.randint(10, size=(data_size, 1))
    y_data_2 = np.ones((data_size, 1))

    # Создаем метки кластеров: 0-чётное число, 1-нечётное
    for i in range(data_size):
        if x_data_2[i] % 2 == 0:
            y_data_2[i] = 0

    # Проводим разделение исходных данных на обучающую и тестовую выборку
    x_train_2, x_test_2, y_train_2, y_test_2 = train_test_split(x_data_2, y_data_2, test_size=test_size_percent)

    # Создаём модель
    model_2 = Sequential()

    # Добавляем слои
    model_2.add(Dense(5, input_dim=1, activation='relu'))
    model_2.add(Dropout(0.5))
    model_2.add(Dense(3, activation='relu'))
    model_2.add(Dropout(0.5))
    model_2.add(Dense(1, activation='sigmoid'))

    # Компилируем модель
    model_2.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    # Обучаем модель
    log_2 = model_2.fit(x_train_2, y_train_2, epochs=epoch, batch_size=20, verbose=0)

    # Проводим оценку точности
    score_2 = model_2.evaluate(x_test_2, y_test_2, batch_size=20)
    print(score_2)

    # Отображаем на графике ошибку
    fig = plt.figure(figsize=(5, 5))
    fig.suptitle('График ошибки')
    plt.plot(log_2.history['loss'], label='Потери')
    plt.legend()
    plt.grid(True)

    # Проводим предсказание
    x = 3
    print('Вход: ', x)
    pred_2 = model_2.predict([x])
    print('Значение выхода нейронной сети: ', pred_2)
    if pred_2 >= 0.5:
        num_class = 1
    else:
        num_class = 0
    print('Номер класса на основе нейронной сети (0-чётное, 1-нечётное): ', num_class)


print('\nПример 2, кол-во эпох 700')
test_2(700, 500, 0.25)
print('\nПример 2, кол-во исходных данных 1000')
test_2(500, 1000, 0.25)
print('\nПример 2, размер партии для обучения 0.85')
test_2(500, 500, 0.15)
print('\nПример 2, дополнительный скрытый слой на 30 нейронов, кол-во эпох 700')
test_2_1(700, 500, 0.25)
print('\nПример 2, дополнительный скрытый слой на 30 нейронов , кол-во исходных данных 1000')
test_2_1(500, 1000, 0.25)
print('\nПример 2, дополнительный скрытый слой на 30 нейронов , размер партии для обучения 0.85')
test_2_1(500, 500, 0.15)
print('\nПример 3, кол-во эпох 100')
test_3(100, 1000, 0.25)
print('\nПример 3, кол-во исходных данных 500')
test_3(40, 500, 0.25)
print('\nПример 3, размер партии для обучения 0.7')
test_3(40, 1000, 0.3)

plt.show()

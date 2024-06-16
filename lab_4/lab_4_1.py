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

# 1.Парная линейная регрессия
# Формируем исходные данные
x_data = np.random.rand(200)  # Генерируем случайные точки
noise = np.random.normal(0, 0.01, x_data.shape)  # Добавляем шум
y_data = x_data * 0.1 + 0.2 + noise  # Вычисляем y

# Разделение данных на обучающую и тестовую выборку
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.25)

# Построить последовательность
model = Sequential()

# Добавляем к модели полносвязный слой
model.add(Dense(units=1, input_dim=1))  # Ввести одномерные данные вывести одномерные данные

# Компилируем модель. Оптимизатор: sgd; Функция потерь: mse - среднеквадратичная ошибка
model.compile(optimizer='sgd', loss='mse')

# Обучаем сеть
log = model.fit(x_train, y_train, epochs=500, batch_size=20, verbose=0)

# Оцениваем качество модели
err = model.evaluate(x_test, y_test)
print('Ошибка: ', err)

# Строим графики
fig1 = plt.figure(figsize=(5, 5))
fig1.suptitle('Результат решения задачи')
y_pred = model.predict(x_data)
plt.plot(x_data, y_pred, 'r-', lw=3, label='Построенная регрессионная модель')
plt.scatter(x_data, y_data, label='Исходные данные')
plt.legend()

# Отображаем на графике ошибку
fig2 = plt.figure(figsize=(5, 5))
fig2.suptitle('График ошибки')
plt.plot(log.history['loss'], label='Потери')
plt.legend()
plt.grid(True)

# 2.Множественная регрессия
# Формируем исходные данные
x_data_1 = np.random.randint(10, size=(500, 2))  # Генерируем случайные точки
y_data_1 = x_data_1[:, 0] ** 2 + x_data_1[:, 0] * x_data_1[:, 1]  # Вычисляем y

# Разделение данных на обучающую и тестовую выборку
x_train_1, x_test_1, y_train_1, y_test_1 = train_test_split(x_data_1, y_data_1, test_size=0.25)

# Построить последовательную модель
model_1 = Sequential()

# Добавляем к модели слои
model_1.add(Dense(units=20, input_dim=2, activation='relu'))
model_1.add(Dense(units=1, activation='linear'))

# Компилируем модель
model_1.compile(optimizer='adam', loss='mse')

# Обучаем сеть
log_1 = model_1.fit(x_train_1, y_train_1, epochs=500, batch_size=20, verbose=0)

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
x1 = int(input('Введите значение первого признака: '))
x2 = int(input('Введите значение второго признака: '))
pred_1 = model_1.predict([[x1, x2]])
print('Значение функции на основе нейронной сети: ', pred_1)
print('Реальный результат: ', x1 ** 2 + x1 * x2)

# 3.Бинарная классификация. Классификация чётных и нечётных цифр
# Генерируем случайные данные от 0 до 9
x_data_2 = np.random.randint(10, size=(1000, 1))
y_data_2 = np.ones((1000, 1))

# Создаем метки кластеров: 0-чётное число, 1-нечётное
for i in range(1000):
    if x_data_2[i] % 2 == 0:
        y_data_2[i] = 0

# Проводим разделение исходных данных на обучающую и тестовую выборку
x_train_2, x_test_2, y_train_2, y_test_2 = train_test_split(x_data_2, y_data_2, test_size=0.25)

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
log_2 = model_2.fit(x_train_2, y_train_2, epochs=40, batch_size=20, verbose=0)

# Проводим оценку точности
score_2 = model_2.evaluate(x_test_2, y_test_2, batch_size=20)
print(score_2)

# Отображаем на графике ошибку
fig4 = plt.figure(figsize=(5, 5))
fig4.suptitle('График ошибки')
plt.plot(log_2.history['loss'], label='Потери')
plt.legend()
plt.grid(True)

# Проводим предсказание
x = int(input('Введите число:'))
pred_2 = model_2.predict([x])
print('Значение выхода нейронной сети: ', pred_2)
if pred_2 >= 0.5:
    num_class = 1
else:
    num_class = 0
print('Номер класса на основе нейронной сети (0-чётное, 1-нечётное): ', num_class)

# 3.Бинарная классификация. Классификация чётных и нечётных цифр, 2 вариант

# Преобразование данных в категориальную форму
y_data_2 = to_categorical(y_data_2, num_classes=2)

# Проводим разделение исходных данных на обучающую и тестовую выборку
x_train_2_1, x_test_2_1, y_train_2_1, y_test_2_1 = train_test_split(x_data_2, y_data_2, test_size=0.25)

# Создаём модель
model_2_1 = Sequential()

# Добавляем слои
model_2_1.add(Dense(5, input_dim=1, activation='relu'))
model_2_1.add(Dropout(0.5))
model_2_1.add(Dense(3, activation='relu'))
model_2_1.add(Dropout(0.5))
model_2_1.add(Dense(2, activation='softmax'))

# Компилируем модель
model_2_1.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# Обучаем модель
log_2_1 = model_2_1.fit(x_train_2_1, y_train_2_1, epochs=40, batch_size=20, verbose=0)

# Проводим оценку точности
score_2_1 = model_2_1.evaluate(x_test_2_1, y_test_2_1, batch_size=20)
print(score_2_1)

# Отображаем на графике ошибку
fig5 = plt.figure(figsize=(5, 5))
fig5.suptitle('График ошибки')
plt.plot(log_2_1.history['loss'], label='Потери')
plt.legend()
plt.grid(True)

# Проводим предсказание
x = int(input('Введите число:'))
pred_2_1 = model_2_1.predict([x])
print('Значение выхода нейронной сети: ', pred_2_1)
print('Номер класса на основе нейронной сети (0-чётное, 1-нечётное): ', np.argmax(pred_2_1))

# 4.Классификация рукописных цифр от 0 до 10
# Формируем исходные данные, импортируем их из mnist датасета
(x_train_3, y_train_3), (x_test_3, y_test_3) = mnist.load_data()

# Выводим исходные данные
print(y_train_3)
fig6 = plt.figure(figsize=(10, 5))
fig6.suptitle('25 первых элеметов выборки')
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_train_3[i], cmap=plt.cm.binary)

# Формируем модель НС. Flatten-слой на входе, Dense-слой на выходе с функцией активации softmax
model_3 = Sequential([Flatten(input_shape=(28, 28, 1)),
                      Dense(128, activation='relu'),
                      Dense(10, activation='softmax')])

print(model_3.summary())  # Вывод структуры НС в консоль

# Стандартизируем входные значения вектора x
x_train_3 = x_train_3 / 255
x_test_3 = x_test_3 / 255

# Преобразуем выходные значения вектора y в категориальную форму
y_train_3_cat = to_categorical(y_train_3, 10)
y_test_3_cat = to_categorical(y_test_3, 10)

# Компилируем модель
model_3.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Обучаем НС
log_3 = model_3.fit(x_train_3, y_train_3_cat, batch_size=32, epochs=10, validation_split=0.2)

# Оцениваем качество модели
err_3 = model_3.evaluate(x_test_3, y_test_3_cat)
print('Значение ошибки: ', err_3)

# Выводим график ошибки
fig7 = plt.figure(figsize=(5, 5))
fig7.suptitle('График ошибки')
plt.plot(log_3.history['loss'], label='Потери')
plt.legend()
plt.grid(True)

# Проверям правильность распознования цифр
fig8 = plt.figure(figsize=(8, 8))
fig8.suptitle('Проверка на тестовых данных')
for n in range(10):
    x = np.expand_dims(x_test_3[n], axis=0)
    res = model_3.predict(x)
    plt.subplot(5, 2, n + 1)
    plt.xticks([])
    plt.yticks([])
    plt.title(f'Тестовая цифра {n+1}, \nрезультат классификации: {np.argmax(res)}')
    plt.imshow(x_test_3[n], cmap=plt.cm.binary)
    print(f'{n+1}. Результат классификации: ', res)
    print('По результатам классификации это цифра: ', np.argmax(res))

# Проверям правильность распознования цифр без вывода изображений
pred_3 = model_3.predict(x_test_3)
pred_3 = np.argmax(pred_3, axis=1)
print(pred_3[:30])
print(y_test_3[:30])

plt.show()


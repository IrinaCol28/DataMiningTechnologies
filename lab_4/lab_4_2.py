import keras
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn import preprocessing

matplotlib.use('TkAgg')

# Множественная регрессия x+y*x-z^2
# Формируем исходные данные
x_data = np.random.randint(10, size=(1000, 3))  # Генерируем случайные точки
y_data = x_data[:, 0] + x_data[:, 1] * x_data[:, 0] - x_data[:, 2] ** 2  # Вычисляем y

# Разделение данных на обучающую и тестовую выборку
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3)

# Стандартизируем данные
scaler_x = preprocessing.StandardScaler()
scaler_y = preprocessing.StandardScaler()
scaler_x.fit(x_data)
scaler_y.fit(y_data.reshape(-1, 1))

x_train_scaled = scaler_x.transform(x_train)
x_test_scaled = scaler_x.transform(x_test)
y_train_scaled = scaler_y.transform(y_train.reshape(-1, 1))
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1))

# Построить последовательную модель
model = Sequential()

# Добавляем к модели слои
model.add(Dense(units=30, input_dim=3, activation='relu'))
model.add(Dense(units=1, activation='linear'))

# Компилируем модель
model.compile(optimizer='adam', loss='mse')

# Обучаем сеть
log = model.fit(x_train_scaled, y_train_scaled, epochs=500, batch_size=15, verbose=0)

# Оцениваем качество модели
err = model.evaluate(x_test_scaled, y_test_scaled, batch_size=15)
print('Ошибка: ', err)

# Отображаем на графике ошибку
fig = plt.figure(figsize=(5, 5))
fig.suptitle('График ошибки')
plt.plot(log.history['loss'], label='Потери')
plt.legend()
plt.grid(True)

# Проводим предсказание
x = int(input('Введите значение первого признака: '))
y = int(input('Введите значение второго признака: '))
z = int(input('Введите значение третьего признака: '))
pred = model.predict(scaler_x.transform([[x, y, z]]))
print('Значение функции на основе нейронной сети: ', scaler_y.inverse_transform(pred))
print('Реальный результат: ', x + y * x - z ** 2)

plt.show()

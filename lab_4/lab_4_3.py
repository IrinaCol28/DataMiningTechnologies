import keras
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical
import pandas as pd
from sklearn import preprocessing

matplotlib.use('TkAgg')


def diabet_var_1(number: int = None):
    df = pd.read_csv('diabetes.csv')
    print(df)

    y_data = df[['Outcome']]
    x_data = df[['Children', 'Glucose', 'BloodPressure', 'SkinThickness',
                 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]

    # Преобразование целевых переменных в бинарный формат
    y_data_categorical = to_categorical(y_data, num_classes=2)

    # Проводим разделение исходных данных на обучающую и тестовую выборку
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data_categorical, test_size=0.20)

    # Нормализуем данные
    scaler_x = preprocessing.MinMaxScaler()
    scaler_x.fit(x_data)

    x_train_scaled = scaler_x.transform(x_train)
    x_test_scaled = scaler_x.transform(x_test)
    print(x_test_scaled[0])
    print(y_test[:4])

    # Создаём модель
    model = Sequential()

    # Добавляем слои
    model.add(Dense(units=16, input_dim=8, activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(units=16, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=2, activation='softmax'))

    # Компилируем модель
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Обучаем модель
    log = model.fit(x_train_scaled, y_train, epochs=100, batch_size=20, verbose=1, validation_split=0.15)

    # Проводим оценку точности
    score = model.evaluate(x_test_scaled, y_test, batch_size=20)
    print('Ошибка модели: ', score)

    # Отображаем на графике ошибку
    fig = plt.figure(figsize=(7, 5))
    fig.suptitle('График ошибки')
    plt.plot(log.history['loss'], label='Потери')
    plt.plot(log.history['val_loss'], label='Валидация потерь')
    plt.legend()
    plt.grid(True)

    # Отображаем на графике точность
    fig2 = plt.figure(figsize=(7, 5))
    fig2.suptitle('График точности')
    plt.plot(log.history['accuracy'], label='Точность')
    plt.plot(log.history['val_accuracy'], label='Валидация точности')
    plt.legend()
    plt.grid(True)

    # Проводим предсказание
    if number is not None:
        x = number
    else:
        x = int(input('Введите число от 0 до 767:'))
    pred_data = scaler_x.transform(x_data.iloc[[x]])
    pred = model.predict(pred_data)
    print('Выбранный пациент: \n', df.iloc[[x]])
    print('Значение выхода нейронной сети: ', pred)
    print('Номер класса на основе нейронной сети (0-болен, 1-здоров): ', np.argmax(pred, axis=1))
    print('Номер класса в исходных данных (0-болен, 1-здоров): ', y_data.at[x, 'Outcome'])

    # Проверям правильность предсказания для 50-ти пациентов
    pred = model.predict(x_test_scaled)
    pred = np.argmax(pred, axis=1)
    print(pred[:30], ' Результаты классификации нейросетью')
    print(np.argmax(y_test[:30], axis=1), ' Исходные данные')
    return score


# diabet_var_1()
# plt.show()

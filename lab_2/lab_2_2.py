import matplotlib
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn import metrics
import numpy as np
# Игнорирование FutureWarning
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

matplotlib.use('TkAgg')

# Функция для получения результатов кластеризации методом k-средних
def calculate_all_predictions(data, cluster_count):
    mod = KMeans(n_clusters=cluster_count)
    mod.fit(data)
    all_pred = mod.predict(data)
    return all_pred


# 1. Подготовливаем данные для кластеризации
# Загрузка данных из csv-файла
df = pd.read_csv("Абоненты.csv")
print(df)


# Формирование датафрейма для кластеризации, удаление ненужных столбцов
df.drop(columns=["Consumption", "Day_calls", "Evening_calls", "Night_calls", "Intercity_calls",
                 "International_calls", "Internet"], inplace=True)
print(df)

# Извлечение данных как массива
mas = df.to_numpy()

# Стандартизируем данные
scaler = preprocessing.StandardScaler()
mas_std = scaler.fit_transform(mas)

print(mas)
print(mas_std)

# Построени точечного графика по признакам Age и Call_duration
fig1 = plt.figure(figsize=(7, 5))
fig1.suptitle('Исходные данные')
plt.xlabel("Age")
plt.ylabel("Call_duration")
plt.scatter(df.Age, df.Call_duration)

# 2. Проводим кластеризацию абонентов иерархическим методом с использованием библиотеки Scipy
# Проводим кластеризацию
fig2 = plt.figure(figsize=(12, 5))
fig2.suptitle('Кластеризация иерархичесим методом')
mergers = linkage(mas_std, method="complete")
dendrogram(mergers, labels=df.index, leaf_font_size=10, leaf_rotation=90)

# 3. Проводим кластеризацию абонентов методом k-means с использованием библиотеки Scicit-learn
# Описываем модель
model = KMeans(n_clusters=5)

# Проводим кластеризацию
model.fit(mas_std)
all_predictions = model.predict(mas_std)

# Выводим результаты кластеризации (метки кластеров, центры кластеров)
print(model.labels_)
print(model.cluster_centers_)

# Визуализируем результаты
fig3 = plt.figure(figsize=(7, 5))
fig3.suptitle('Разбиение на 5 кластеров (метод k-средних)')
plt.xlabel("Age")
plt.ylabel("Call_duration")
plt.scatter(df.Age, df.Call_duration, c=all_predictions)

# Добавление столбца Clusters
df["Clusters"] = all_predictions
print("\n", df.sort_values(by="Clusters"))

# Оценка качества кластеризации
krit = metrics.calinski_harabasz_score(mas_std, all_predictions)
print(f"\nКритерий отношения дисперсии исходной модели: {krit}\n")

# Вычисление наилучшего кол-ва кластеров
clusters = np.arange(2, 8)
for cluster in clusters:
    print(f"Критерий отношения дисперсии при {cluster} кластерах:",
          metrics.calinski_harabasz_score(mas_std, calculate_all_predictions(mas_std, cluster)))

# Визуализация оптимального варианта
fig4 = plt.figure(figsize=(7, 5))
fig4.suptitle('Оптимальный вариант разбиения на 6 кластеров (метод k-средних)')
plt.xlabel("Age")
plt.ylabel("Call_duration")
plt.scatter(df.Age, df.Call_duration, c=calculate_all_predictions(mas_std, 6))


plt.show()

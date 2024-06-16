import matplotlib
import pandas as pd
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn import metrics
# Игнорирование FutureWarning
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
matplotlib.use('TkAgg')

#считываем данные из файла
df = pd.read_csv('Iris.csv')
print('Получаем датафрейм следующего вида:')
print(df)

#построим график для двух параметров
fig1=plt.figure(figsize=(7, 5))
plt.xlabel('Длина чашелистника')
plt.ylabel('Ширина чашелистника')
plt.scatter(df.SepalLengthCm, df.SepalWidthCm)
plt.title('Ирисы Фишера')

#создадим копию датафрейма
dff = df.copy()

#удалим последний столбец
dff.pop('Species')

#Извлекаем из датафрейма измерения как массив
samples = dff.values

#Описываем модель (задаём число кластеров, равное 3)
model = KMeans(n_clusters=3)

#Проводим кластеризацию
model.fit(samples)

#Выводим результаты кластеризации
all_predictions = model.predict(samples)
print('Результаты кластеризации:')
print(all_predictions)

#Проводим визуализацию, выделяя разными цветами точки, попавшие вразные кластеры
fig2=plt.figure(figsize=(7, 5))
plt.xlabel('Длина чашелистника')
plt.ylabel('Ширина чашелистника')
plt.title('Кластеризация')
plt.scatter(df.SepalLengthCm, df.SepalWidthCm, c=all_predictions)


#добавим столбец Clusters с результатами кластеризации
df['Clusters'] = all_predictions
print(df.to_string())

#преобразуем столбец Speties в категориальную переменную
df['Species'] = pd.factorize(df['Species'])[0]

#Сравним два последних столбца датафрейма
print('Датафрейм с добавленными столбцами')
print(df.to_string())

print('Статистика:')
#получение статистики
print(df.groupby(['Clusters', 'Species']).count())

#сводная таблица
print(df.groupby(['Clusters', 'Species'])['Clusters'].count().unstack())

# Оценка качества кластеризации
krit = metrics.calinski_harabasz_score(samples, all_predictions)
print(f"\nОценка качества кластеризации: {krit}\n")






#СТАНДАРТИЗАЦИЯ

#считываем данные из файла
df = pd.read_csv('Iris.csv')
print('Получаем датафрейм следующего вида:')
print(df)

#создадим копию датафрейма
dff = df.copy()

#удалим последний столбец
dff.pop('Species')

#Извлекаем из датафрейма измерения как массив
samples = dff.values

#Выполняем стандартизацию
scaler=preprocessing.StandardScaler()
samples_std=scaler.fit_transform(samples)

#Описываем модель (задаём число кластеров, равное 3)
model = KMeans(n_clusters=3)

#Проводим кластеризацию
model.fit(samples_std)

#Выводим результаты кластеризации
all_predictions = model.predict(samples_std)
print('Результаты кластеризации:')
print(all_predictions)

#Проводим визуализацию, выделяя разными цветами точки, попавшие вразные кластеры
fig3=plt.figure(figsize=(7, 5))
plt.xlabel('Длина чашелистника')
plt.ylabel('Ширина чашелистника')
plt.title('Кластеризация для стандартизированных данных')
plt.scatter(df.SepalLengthCm, df.SepalWidthCm, c=all_predictions)

#добавим столбец Clusters с результатами кластеризации
df['Clusters'] = all_predictions
print(df.to_string())

#преобразуем столбец Speties в категориальную переменную
df['Species'] = pd.factorize(df['Species'])[0]

#Сравним два последних столбца датафрейма
print('Датафрейм с добавленными столбцами')
print(df.to_string())

print('Статистика для стандартизированных данных:')
#получение статистики
print(df.groupby(['Clusters', 'Species']).count())

#сводная таблица
print(df.groupby(['Clusters', 'Species'])['Clusters'].count().unstack())

# Оценка качества кластеризации
krit = metrics.calinski_harabasz_score(samples_std, all_predictions)
print(f"\nОценка качества кластеризации: {krit}\n")

plt.show()
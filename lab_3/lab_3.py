# 1.Импорт необходимых библиотек
import matplotlib
import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels import tsa
from statsmodels import graphics

matplotlib.use('TkAgg')


# Метод для вывода на экран результатов теста Дики-Фуллера
def ad_fuller_result(data):
    result = tsa.stattools.adfuller(data)
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    if result[1] > 0.05:
        print('Ряд не стационарный')
    else:
        print('Ряд стационарный')


# 2.Загрузка датасета
# Загрузка датасета, как временного ряда
df = pd.read_csv("stock_data.csv", parse_dates=["Date"], index_col='Date')

print(df)

df.info()

# 3.Первичный анализ временного ряда
print(df.describe())

# Для прогнозирования возьмем самую низкую за день цену Low
df_low = df[["Low"]]

print(df_low)

# Строим график
fig1 = plt.figure(figsize=(10, 5))
fig1.suptitle('Исходные данные')
df_low.Low.plot()

# Агрегируем данные по месяцам и используем среднемесячные показатели
df_low = df_low.resample('M').mean()
print(df_low)

# Строим график агрегированных данных
fig2 = plt.figure(figsize=(10, 5))
fig2.suptitle('Агрегированные по месяцам данные')
df_low.Low.plot()

# Строим столбиковую диаграмму с агрегированием по годам
df_low.resample('Y').mean().plot(kind='bar', title="Агрегированные по годам данные", figsize=(5, 5))

# 4.Исследование структуры временного ряда
# Проводим разложение временного ряда
result_add = tsa.seasonal.seasonal_decompose(df_low, model='additive', extrapolate_trend='freq')
result_add.plot().suptitle("Additive Decompose")

# Извлекаем компоненты
df_reconstructed = pd.concat([result_add.seasonal, result_add.trend, result_add.resid, result_add.observed], axis=1)
df_reconstructed.columns = ['Seasonality', 'Trend', 'Residual', 'Actual_values']
print(df_reconstructed.head())

# 5.Проверка гипотезы о стационарности временного ряда
ad_fuller_result(df_low['Low'])

# Ряд не стационарный, переходим к разностям
df_low_diff = df_low.diff().dropna()
df_low_diff.plot(title="Дифференцированный ряд")
ad_fuller_result(df_low_diff['Low'])

# 6.Определение параметров модели ARIMA
# Строим кореллограмму
fig3 = plt.figure(figsize=(12, 8))
fig3.suptitle('Кореллограммы', fontsize=16)
ax1 = fig3.add_subplot(211)
fig3 = graphics.tsaplots.plot_acf(df_low_diff, lags=25, ax=ax1)
ax2 = fig3.add_subplot(212)
fig3 = graphics.tsaplots.plot_pacf(df_low_diff, lags=25, method='ywm', ax=ax2)

# 7.Построение модели ARIMA
# Создание объекта моделирования
model = sm.tsa.ARIMA(df_low, order=(2, 1, 2))

# Проводим моделирование
result = model.fit()

print(result.summary())

# Изменяем параметры модели, т.к. не все коэффициенты являются значимыми
# Создание объекта моделирования
model2 = sm.tsa.ARIMA(df_low, order=(1, 2, 1))

# Проводим моделирование
result = model2.fit()

print(result.summary())

# 8.Оценка качества модели
df_res = df_low.copy()
df_res['Forecast'] = result.fittedvalues
df_res['Err'] = df_res['Low'] - df_res['Forecast']
df_res['Err_percent'] = abs((df_res['Err'] / df_res['Low']) * 100)
print(df_res)

# Построение графика для сравнения результатов моделирования
df_res[['Forecast', 'Low']].plot(title="Сравнение результатов моделирования", figsize=(10, 5))

# Количественная оценка качества полученной модели по MAE, MSE, RMSE
print('mae=', result.mae)
print('mse=', result.mse)
print('rmse=', result.mse ** (1 / 2))
mae = abs(df_res['Err']).mean()
print('Проверка mae=', mae)
mse = abs(df_res['Err'] ** 2).mean()
print('Проверка mse=', mse)
rmse = mse ** (1 / 2)
print('Проверка rmse=', rmse)
# Оценка по MAPE
mape = df_res.Err_percent.mean()
print('Средняя ошибка аппроксимации: ', mape)

# 9.Прогнозирование
pred = result.predict('2006-1', '2018-12')
print(pred)
fig4 = plt.figure(figsize=(12, 8))
fig4.suptitle('Прогнозирование')
plt.plot(pred, label='Спрогнозированные значения')
plt.plot(df_low.Low, label='Исходные значения')
plt.legend()


plt.show()

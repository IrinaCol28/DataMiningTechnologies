import matplotlib
import matplotlib.pyplot as plt

from lab_4.lab_4_3 import diabet_var_1
from lab_4.lab_4_3_1 import diabet_var_2
from lab_4.lab_4_3_2 import diabet_var_3
import numpy as np

matplotlib.use('TkAgg')


num_tests = 20
results_1 = []
results_2 = []
results_3 = []

for _ in range(num_tests):
    score_1 = diabet_var_1(123)
    score_2 = diabet_var_2(123)
    score_3 = diabet_var_3(123)

    results_1.append(score_1)
    results_2.append(score_2)
    results_3.append(score_3)

average_result_1 = np.mean(results_1, axis=0)
average_result_2 = np.mean(results_2, axis=0)
average_result_3 = np.mean(results_3, axis=0)

print(f'Средний результат 1-го варианта нейросети: {average_result_1}')
print(f'Средний результат 2-го варианта нейросети: {average_result_2}')
print(f'Средний результат 3-го варианта нейросети: {average_result_3}')

# score_1 = diabet_var_1(123)
# score_2 = diabet_var_2(123)
# score_3 = diabet_var_3(123)
#
# print('Ошибка и точность 1-го варианта нейросети: ', score_1)
# print('Ошибка и точность 2-го варианта нейросети: ', score_2)
# print('Ошибка и точность 3-го варианта нейросети: ', score_3)
#
# plt.show()

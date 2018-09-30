import matplotlib.pyplot as plt
import numpy as np

plt.style.use('bmh')

K = 1000  # число реализаций Гауссовской случайной величины (СВ)
n = 12  # число реализаций равномерной случайной величины для генерации одной реализации гауссовской СВ
mu = 1
sig = 2

# Теоретические значения мат.ожидания и дисперсии из таблицы
m = mu
d = sig ** 2

# Генерация alpha (равномерной СВ) и реализаций гауссовской СВ
alf = np.random.rand(n, K)  # матрица из K столбцов по n элементов
x = mu + sig * (np.sum(alf, 0) - 6)  # сумма по столбцам матрицы alf<=p

## 2. Вычисление выборочных дисперсий
ds = np.zeros(K)
for k in range(1, K):
    ds[k] = np.var(x[0: k])

plt.plot(np.arange(0, K), ds)
plt.plot(np.arange(0, K), d * np.ones(K))
plt.title("Дисперсия")
plt.show()

## 3. Вычисление выборочных матожиданий
ms = np.zeros(K)
for k in range(1, K):
    ms[k] = np.average(x[0: k])

plt.figure()
plt.plot(np.arange(0, K), ms)
plt.plot(np.arange(0, K), m * np.ones(K))
plt.title("Матожидание")
plt.show()

## 4. Плотность и гистограмма
t = np.arange(np.min(x), np.max(x), 0.05)     # упорядоченный массив возможных значений случайной величины
px = 1/(sig*np.sqrt(2*np.pi))*np.exp(-0.5*(t-mu)**2/sig**2)  # формула плотности из таблицы


plt.figure()
plt.hist(x,t, density=True, histtype='bar')
plt.plot(t, px, 'g')   # отрисовка графика плотности
plt.title("Плотность и возможные значение")
plt.show()
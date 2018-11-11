# 4.
# Вычислить абсолютную ошибку оценивания плотности распределения случайного вектора в двумерном пространстве признаков при использовании оценки Парзена.
# Построить график зависимости ошибки оценивания от величины параметра оконной функции.
# Используйте одну из следующих оконных функций:
# a.гауссовская функция c использованием диагональной матрицы;
# b.гауссовская функция c использованием матрицы ковариаций;
# c.показательная функция;
# d.оконная прямоугольная функция;
# e.оконная треугольная функция.

import operator

import matplotlib.pyplot as plt
import numpy as np
import prettytable


def pdf_multivariate_gauss(x_samples, mu, cov):
    part1 = 1 / (((2 * np.pi) ** (len(mu) / 2)) * (np.linalg.det(cov) ** (1 / 2)))
    part2 = (-1 / 2) * ((x_samples - mu).T.dot(np.linalg.inv(cov))).dot((x_samples - mu))
    return float(part1 * np.exp(part2))


def parzen_window_est(x_samples, h=1, center=[0, 0]):
    dimensions = x_samples.shape[1]

    k = 0
    for x in x_samples:
        is_inside = 1
        for axis, center_point in zip(x, center):
            if np.abs(axis - center_point) > (h / 2):
                is_inside = 0
        k += is_inside
    return (k / len(x_samples)) / (h ** dimensions)


samplesNumber = 10000
mu_vector = np.array([0, 0])
cov_matrix = np.array([[1, 0], [0, 1]])
x = np.random.multivariate_normal(mu_vector, cov_matrix, samplesNumber)

# generate a range of 400 window widths between 0 < h < 1
h_range = np.linspace(0.001, 1, 400)

# calculate the actual density at the center [0, 0]
mu_vector = np.array([[0], [0]])
cov_matrix = np.eye(2)
actual_pdf_val = pdf_multivariate_gauss(np.array([[0], [0]]), mu_vector, cov_matrix)

# get a list of the differnces (|estimate-actual|) for different window widths
parzen_estimates = [np.abs(parzen_window_est(x, h=i, center=[0, 0])
                           - actual_pdf_val) for i in h_range]

plt.plot(h_range, parzen_estimates)
plt.xlabel("Window size. h")
plt.ylabel("Absolute error")
plt.show()

# get the window width for which |estimate-actual| is closest to 0
min_index, min_value = min(enumerate(parzen_estimates), key=operator.itemgetter(1))

optimal_h = h_range[min_index]
print('Optimal window width for this data set: ', optimal_h)

# test on some points
p1 = parzen_window_est(x, h=optimal_h, center=[0, 0])
p2 = parzen_window_est(x, h=optimal_h, center=[0.3, 0.3])
p3 = parzen_window_est(x, h=optimal_h, center=[0.7, 0.5])

mu = np.array([[0], [0]])
cov = np.eye(2)

a1 = pdf_multivariate_gauss(np.array([[0], [0]]), mu, cov)
a2 = pdf_multivariate_gauss(np.array([[0.3], [0.3]]), mu, cov)
a3 = pdf_multivariate_gauss(np.array([[0.7], [0.5]]), mu, cov)

results = prettytable.PrettyTable(["", "predicted", "actual"])
results.add_row(["p([0,0]^t", p1, a1])
results.add_row(["p([0.3,0.3]^t", p2, a2])
results.add_row(["p([0.7,0.5]^t", p3, a3])

print(results)

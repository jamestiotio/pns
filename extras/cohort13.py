#%%
import numpy as np

#%%
# Example 1
x = np.arange(1.0, 8 + 1, 1.0)
y = np.array([6.9, 10.8, 9.3, 7.8, -0.7, -9.2, -22.1, -37.7])

x_mean = np.mean(x)
y_mean = np.mean(y)

b1 = np.dot(y_mean - y, x_mean - x) / np.sum(np.power(x_mean - x, 2))
b0 = y_mean - b1 * x_mean
sigma_squared = (
    1 / x.shape[0] * np.sum([(yi - b0 - b1 * xi) ** 2 for (xi, yi) in zip(x, y)])
)
print(f"b0 = {b0}")
print(f"b1 = {b1}")
print(f"sigma_squared = {sigma_squared}")

# %%
# Example 2
x = np.array([1.0, 3.0, 6.0, 10.0])
y = np.array([1.1, 2.9, 6.1, 10.1])

x_mean = np.mean(x)
y_mean = np.mean(y)

b1 = np.dot(y_mean - y, x_mean - x) / np.sum(np.power(x_mean - x, 2))
b0 = y_mean - b1 * x_mean
sigma_squared = (
    1 / x.shape[0] * np.sum([(yi - b0 - b1 * xi) ** 2 for (xi, yi) in zip(x, y)])
)
print(f"b0 = {b0}")
print(f"b1 = {b1}")
print(f"sigma_squared = {sigma_squared}")

# %%
# Example 3
T_denominator = np.sqrt(np.sum([(yi - b0 - b1 * xi) ** 2 for (xi, yi) in zip(x, y)]))
print(T_denominator)

sigma_prime = np.sqrt(
    (
        1
        / (x.shape[0] - 2)
        * np.sum([(yi - b0 - b1 * xi) ** 2 for (xi, yi) in zip(x, y)])
    )
)
sx = np.sqrt(np.sum([(xi - x_mean) ** 2 for xi in x]))
b0_distribution_denominator = sigma_prime * np.sqrt(
    1 / x.shape[0] + x_mean ** 2 / sx ** 2
)
print(b0_distribution_denominator)
print(T_denominator / b0_distribution_denominator)

T = (b0 - 0.448) / T_denominator
print(T)
print(T * T_denominator / b0_distribution_denominator)

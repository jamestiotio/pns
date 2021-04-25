#%%
import numpy as np

#%%
# Example 2
x = np.arange(1.0, 8 + 1, 1.0)
y = np.array([6.9, 10.8, 9.3, 7.8, -0.7, -9.2, -22.1, -37.7])
#%%
# Do linear regression with lecture method
x_mean = np.mean(x)
y_mean = np.mean(y)

m = np.dot(y_mean - y, x_mean - x) / np.sum(np.power(x_mean - x, 2))
c = y_mean - m * x_mean
print(f"r = {m}")
print(f"n0 = {np.exp(c)}")

#%%
# Double check with linear model from sklearn
from sklearn import linear_model

model = linear_model.LinearRegression()
model.fit(x.reshape(-1, 1), y)
print(f"r = {model.coef_[0]}")
print(f"n0 = {np.exp(model.intercept_)}")

# %%
# Solve with minimise S method
b_0_coef_1 = -2 * -x.shape[0]
b_1_coef_1 = -2 * -np.sum(x)
constant_1 = -2 * np.sum(y)

b_0_coef_2 = -2 * -np.sum(x)
b_1_coef_2 = -2 * -np.sum(x ** 2)
constant_2 = -2 * x @ y

print(f"{b_0_coef_1}*b0 + {b_1_coef_1}*b1 + {constant_1} = 0")
print(f"{b_0_coef_2}*b0 + {b_1_coef_2}*b1 + {constant_2} = 0")

np.linalg.solve(
    np.array([[b_0_coef_1, b_1_coef_1], [b_0_coef_2, b_1_coef_2]]),
    np.array([-constant_1, -constant_2]),
)
# %%
# %%
# Example 3
# Solve with minimise S method 3 var
b_0_coef_1 = -2 * -x.shape[0]
b_1_coef_1 = -2 * -np.sum(x)
b_2_coef_1 = -2 * -np.sum(x ** 2)
constant_1 = -2 * np.sum(y)

b_0_coef_2 = -2 * -np.sum(x)
b_1_coef_2 = -2 * -np.sum(x ** 2)
b_2_coef_2 = -2 * -np.sum(x ** 3)
constant_2 = -2 * x @ y

b_0_coef_3 = -2 * -np.sum(x ** 2)
b_1_coef_3 = -2 * -np.sum(x ** 3)
b_2_coef_3 = -2 * -np.sum(x ** 4)
constant_3 = -2 * x ** 2 @ y

print(f"{b_0_coef_1}*b0 + {b_1_coef_1}*b1 + {b_2_coef_1}*b2 + {constant_1} = 0")
print(f"{b_0_coef_2}*b0 + {b_1_coef_2}*b1 + {b_2_coef_2}*b2 + {constant_2} = 0")
print(f"{b_0_coef_3}*b0 + {b_1_coef_3}*b1 + {b_2_coef_3}*b2 + {constant_3} = 0")

np.linalg.solve(
    np.array(
        [
            [b_0_coef_1, b_1_coef_1, b_2_coef_1],
            [b_0_coef_2, b_1_coef_2, b_2_coef_2],
            [b_0_coef_3, b_1_coef_3, b_2_coef_3],
        ]
    ),
    np.array([-constant_1, -constant_2, -constant_3]),
)
# %%

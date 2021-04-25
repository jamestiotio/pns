# Naive Implementation of SUTD ISTD 2021 50.034 Introduction to Probability and Statistics Midterm Exam Final Question
# Created by James Raphael Tiovalen (2021)

import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng()

LAMBDA = rng.uniform(0, 1)
N = 1_000_000
DATA_POINTS = 1_000_000
BINS = 200

# Assume that each call to rand() (i.e., each sample drawn) is independent and identically distributed (i.i.d.)
def rand(num_of_samples):
    return rng.uniform(-LAMBDA, LAMBDA, num_of_samples)


# As N -> +∞, this becomes approximately closer and closer to a normal distribution with mean 0 and variance (N * (LAMBDA ** 2)) / 3
# These properties actually agree with the reformulated/reinterpreted Central Limit Theorem approximation in terms of sums of i.i.d. random variables
def gauss():
    return np.sum(rand(N))


def main():
    nums = np.array([gauss() for _ in np.arange(DATA_POINTS)])
    actual = rng.normal(0, (LAMBDA * np.sqrt(N / 3)), DATA_POINTS)

    # Capture and show all data points within three standard deviations from the mean (≈99.73%)
    bins = np.linspace(
        -3 * (LAMBDA * np.sqrt(N / 3)), 3 * (LAMBDA * np.sqrt(N / 3)), BINS
    )

    # Plot the collected data points to illustrate the normal distribution shape and compare with the actual normal distribution
    plt.style.use("seaborn-deep")
    plt.hist(nums, bins, alpha=0.5, label="nums")
    plt.hist(actual, bins, alpha=0.5, label="actual")
    plt.legend(loc="upper right")
    plt.show()


if __name__ == "__main__":
    main()
# Main correlation finder helper tool
# Created by Velusamy Sathiakumar Ragul Balaji (2021)
# Improved by James Raphael Tiovalen (2021)

# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# The stronger the correlation, the better and more convincing the results are/will be
THRESHOLD = 0.7

SELECTED_DATASET = ""

# This algorithm is general (should fit any dataset of choice)
def main():
    df = pd.read_csv(SELECTED_DATASET)
    corr = df.corr(method="pearson")
    mask = np.ones_like(corr)
    corr_matrix = corr.to_dict()

    # For debugging purposes
    print(corr)

    # Finder/selector of 3 random variables with 2 +ve correlations and 1 -ve correlation
    for i_idx, i in enumerate(corr_matrix):
        for j_idx, j in enumerate(corr_matrix[i]):
            if i == j:
                continue
            if corr_matrix[i][j] < -THRESHOLD:
                for k_idx, k in enumerate(corr_matrix):
                    if i == k or j == k:
                        continue
                    if corr_matrix[i][k] > THRESHOLD and corr_matrix[k][j] > THRESHOLD:
                        # Only select the three relevant random variables to be plotted
                        mask[i_idx, k_idx], mask[k_idx, j_idx], mask[i_idx, j_idx] = (
                            False,
                            False,
                            False,
                        )
                        print(
                            i,
                            k,
                            j,
                            corr_matrix[i][k],
                            corr_matrix[k][j],
                            corr_matrix[i][j],
                        )
                        # Plot only our selected correlations
                        sns.heatmap(
                            corr,
                            mask=mask,
                            xticklabels=corr.columns.values,
                            yticklabels=corr.columns.values,
                        )
                        plt.show()
                        # Reset mask matrix
                        mask[i_idx, k_idx], mask[k_idx, j_idx], mask[i_idx, j_idx] = (
                            True,
                            True,
                            True,
                        )


if __name__ == "__main__":
    main()
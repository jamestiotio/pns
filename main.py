# Main correlation finder helper tool
# Created by Velusamy Sathiakumar Ragul Balaji (2021)
# Improved by James Raphael Tiovalen (2021)

# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse


# This algorithm is general (should fit any CSV dataset of choice)
# This implementation will print exactly twice as many desired tri-correlation relationships as we want (duplicates at upper triangle and lower triangle of the matrix), but we don't really care since we can pick, choose and sieve out manually by ourselves for the sake of this project
def main():
    # Initialize argument parser
    parser = argparse.ArgumentParser(
        description="Conveniently select appropriate/relevant triplets of random variables to prove the non-transitivity property of Pearson's correlation coefficient.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-f",
        "--file",
        "--dataset",
        "--csv",
        type=str,
        nargs=1,
        help="the CSV dataset input file to be processed",
        required=True,
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        nargs="?",
        const=0.7,
        default=0.7,
        help="the threshold for the correlation coefficient strength to be considered/taken into account",
        required=False
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        nargs=1,
        default=None,
        help="the base non-indexed output image filename to save the correlation matrix plot(s) to",
        required=False
    )
    args = vars(parser.parse_args())

    # The stronger the correlation, the better and more convincing the results are/will be
    THRESHOLD = args["threshold"]

    file_idx = 0

    # Start algorithm
    SELECTED_DATASET = args["file"][0]
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

                        # Decide whether to display/show image plot to screen or save image plot to file
                        if args["output"] is None:
                            plt.show()
                        else:
                            filename = args["output"][0].rpartition(".")
                            plt.savefig(filename[0] + str(file_idx) + "." + filename[2], bbox_inches="tight")
                            file_idx += 1

                        # Reset and clear the plot
                        plt.clf()

                        # Reset mask matrix
                        mask[i_idx, k_idx], mask[k_idx, j_idx], mask[i_idx, j_idx] = (
                            True,
                            True,
                            True,
                        )


if __name__ == "__main__":
    main()
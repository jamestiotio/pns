# SUTD ISTD 2021 50.034 Introduction to Probability and Statistics 1D Project

Topic: Non-Transitivity Property of Pearson's Correlation Coefficient

<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg?style=for-the-badge"></a>

Team Members:

- [Velusamy Sathiakumar Ragul Balaji](https://github.com/ragulbalaji)
- [James Raphael Tiovalen](https://github.com/jamestiotio)
- [Shoham Chakraborty](https://github.com/shohamc1)

## Repository Details

This repository houses the code scripts, as well as the datasets that we utilized in this short research as we attempt to find a real-world/real-life example/sample to showcase the non-transitivity property of Pearson's correlation coefficient between 3 random variables.

## Usage

These are the available flags and options:

```console
$ python3 main.py --help
Usage: main.py [-h] -f FILE [-t [THRESHOLD]] [-o OUTPUT]

Conveniently select appropriate/relevant triplets of random variables to prove
the non-transitivity property of Pearson's correlation coefficient.

Options:
  -h, --help            show this help message and exit
  -f FILE, --file FILE, --dataset FILE, --csv FILE
                        the CSV dataset input file to be processed (default:
                        None)
  -t [THRESHOLD], --threshold [THRESHOLD]
                        the threshold for the correlation coefficient strength
                        to be considered/taken into account (default: 0.7)
  -o OUTPUT, --output OUTPUT
                        the base non-indexed output image filename to save the
                        correlation matrix plot(s) to (default: None)
```

## Acknowledgements

- IEEE dataset obtained from [this research paper](https://ieeexplore.ieee.org/document/6862882).
- Top 100 most valuable GitHub repositories list obtained from [this article](https://hackernoon.com/githubs-top-100-most-valuable-repositories-out-of-96-million-bb48caa9eb0b).
- Kaggle housing prices dataset obtained from either [here](https://www.kaggle.com/c/home-data-for-ml-course/data) or [here](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data) (they have the same training dataset).
- India's graduate admissions dataset obtained from [here](https://www.kaggle.com/mohansacharya/graduate-admissions/data).
- Financial indicators of US stocks (2018) dataset obtained from [here](https://www.kaggle.com/cnic92/200-financial-indicators-of-us-stocks-20142018/data).
- Pokemon dataset obtained from [here](https://www.kaggle.com/mariotormo/complete-pokemon-dataset-updated-090420/data).
- Rolling correlation matrix of the prices of cryptocurrencies over time can be retrieved from [here](https://cryptowat.ch/correlations).
- Latest worldwide COVID-19 per-country statistics (retrieved on 11 March 2021) provided by Our World in Data [here](https://github.com/owid/covid-19-data/tree/master/public/data).
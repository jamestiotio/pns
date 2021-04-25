#!/usr/bin/python3
# SUTD ISTD 2021 50.034 Introduction to Probability and Statistics Utility Class Library
# Created by James Raphael Tiovalen (2021)

from scipy.stats import (
    uniform,
    bernoulli,
    binom,
    geom,
    poisson,
    expon,
    beta,
    gamma,
    pareto,
    norm,
    chi2,
    t,
    ttest_1samp,
    chisquare,
)
from scipy.integrate import quad
import numpy as np
import numpy.polynomial.polynomial as poly


class PNSUtils:
    """
    A class of useful and convenient utilities to be used for SUTD ISTD 50.034: Introduction to Probability and Statistics.
    Generally, the survival function methods of scipy are relatively more accurate sometimes as compared to the actual cumulative distribution function methods.
    Use this utility class to find relevant values of statistics of parameters, especially those with distributions with no closed forms/formulas.
    For discrete cases, they are using exact computations so there is no need for continuity correction (no approximations).
    Please do take note that some methods are only applicable in very specific and narrow scenarios and constraints.
    """

    def __init__(self):
        pass

    def calculate_mean(self, random_sample):
        return np.mean(random_sample)

    def calculate_stddev(self, random_sample, unbiased=False):
        if unbiased:
            return np.std(random_sample, ddof=1)
        else:
            # Default ddof = 0
            return np.std(random_sample)

    def calculate_variance(self, random_sample, unbiased=False):
        if unbiased:
            return np.var(random_sample, ddof=1)
        else:
            # Default ddof = 0
            return np.var(random_sample)

    def draw_from_custom_discrete_rv(self, elements, probabilities, n):
        if (np.array(probabilities) < 0).any() or (np.array(probabilities) > 1).any():
            raise NotImplementedError("Probability values should lie between 0 and 1.")
        elif np.sum(probabilities) != 1:
            raise NotImplementedError(
                "Sum of the probability values should be equal to 1."
            )

        return np.random.choice(elements, n, p=probabilities)

    def draw_from_custom_continuous_rv(self, pdf_func, lower_bound, upper_bound, n):
        # TODO
        ans, err = quad(pdf_func, upper_bound, lower_bound)
        if ans != 1:
            raise NotImplementedError(
                "Integral of the probability values should be equal to 1."
            )
        if (
            pdf_func(lower_bound) < 0
            or pdf_func(upper_bound) < 0
            or pdf_func(np.average([lower_bound, upper_bound])) < 0
        ):
            raise NotImplementedError("Invalid integrand function!")

    def calc_pr_bernoulli(self, p, x1, x2=None):
        """
        Calculates the requested probability value pertaining to the Bernoulli distribution.
        If only one parameter is passed, then this function will return the value of the Bernoulli CDF up to the value of that parameter: Pr(X <= x1).
        If two parameters are passed, then this function will return the value of the sum between those two values: Pr(x1 <= X <= x2).

        Parameters
        ----------
        p: float
            Success parameter of the Bernoulli distribution. Takes on values where 0 <= p <= 1.
        x1: int
            The first threshold value for the probability.
        x2: int, optional
            The second threshold value for the probability.

        Raises
        ----------
        NotImplementedError
            If no parameters are passed in, if p < 0 or p > 1, or if x2 < x1.
        """

        if p is None or x1 is None:
            raise NotImplementedError(
                "At least two parameters need to be passed into this function!"
            )

        elif p < 0 or p > 1:
            raise NotImplementedError("Invalid parameter!")

        elif x2 is not None and x2 < x1:
            raise NotImplementedError(
                "The third parameter should have a value of at least the value of the second parameter!"
            )

        if x2 is None:
            return 1 - bernoulli.sf(x1, p)
        else:
            return bernoulli.sf(x1, p) - bernoulli.sf(x2, p)

    def calc_pr_binom(self, n, p, x1, x2=None):
        """
        Calculates the requested probability value pertaining to the binomial distribution.
        If only one parameter is passed, then this function will return the value of the binomial CDF up to the value of that parameter: Pr(X <= x1).
        If two parameters are passed, then this function will return the value of the sum between those two values: Pr(x1 <= X <= x2).

        Parameters
        ----------
        n: int
            Number of trials parameter of the binomial distribution. Takes on values where n >= 0.
        p: float
            Success parameter of the binomial distribution. Takes on values where 0 <= p <= 1.
        x1: int
            The first threshold value for the probability.
        x2: int, optional
            The second threshold value for the probability.

        Raises
        ----------
        NotImplementedError
            If no parameters are passed in, if n < 0, if p < 0 or p > 1, or if x2 < x1.
        """

        if n is None or p is None or x1 is None:
            raise NotImplementedError(
                "At least three parameters need to be passed into this function!"
            )

        elif n < 0 or p < 0 or p > 1:
            raise NotImplementedError("Invalid parameters!")

        elif x2 is not None and x2 < x1:
            raise NotImplementedError(
                "The fourth parameter should have a value of at least the value of the third parameter!"
            )

        if x2 is None:
            return 1 - binom.sf(x1, n, p)
        else:
            return binom.sf(x1, n, p) - binom.sf(x2, n, p)

    def calc_pr_geom(self, p, x1, x2=None):
        """
        Calculates the requested probability value pertaining to the geometric distribution.
        If only one parameter is passed, then this function will return the value of the geometric CDF up to the value of that parameter: Pr(X <= x1).
        If two parameters are passed, then this function will return the value of the sum between those two values: Pr(x1 <= X <= x2).

        Parameters
        ----------
        p: float
            Success parameter of the geometric distribution. Takes on values where 0 <= p <= 1.
        x1: int
            The first threshold value for the probability.
        x2: int, optional
            The second threshold value for the probability.

        Raises
        ----------
        NotImplementedError
            If no parameters are passed in, if p < 0 or p > 1, or if x2 < x1.
        """

        if p is None or x1 is None:
            raise NotImplementedError(
                "At least two parameters need to be passed into this function!"
            )

        elif p < 0 or p > 1:
            raise NotImplementedError("Invalid parameter!")

        elif x2 is not None and x2 < x1:
            raise NotImplementedError(
                "The third parameter should have a value of at least the value of the second parameter!"
            )

        if x2 is None:
            return 1 - geom.sf(x1, p)
        else:
            return geom.sf(x1, p) - geom.sf(x2, p)

    def calc_pr_poisson(self, lmbd, x1, x2=None):
        """
        Calculates the requested probability value pertaining to the Poisson distribution.
        If only one parameter is passed, then this function will return the value of the Poisson CDF up to the value of that parameter: Pr(X <= x1).
        If two parameters are passed, then this function will return the value of the sum between those two values: Pr(x1 <= X <= x2).

        Parameters
        ----------
        lmbd: float
            The shape parameter of the Poisson distribution. Takes on positive real values.
        x1: int
            The first threshold value for the probability.
        x2: int, optional
            The second threshold value for the probability.

        Raises
        ----------
        NotImplementedError
            If no parameters are passed in, if lmbd <= 0, or if x2 < x1.
        """

        if lmbd is None or x1 is None:
            raise NotImplementedError(
                "At least two parameters need to be passed into this function!"
            )

        elif lmbd <= 0:
            raise NotImplementedError("Invalid parameter!")

        elif x2 is not None and x2 < x1:
            raise NotImplementedError(
                "The third parameter should have a value of at least the value of the second parameter!"
            )

        if x2 is None:
            return 1 - poisson.sf(x1, lmbd)
        else:
            return poisson.sf(x1, lmbd) - poisson.sf(x2, lmbd)

    def calc_pr_expon(self, lmbd, x1, x2=None):
        """
        Calculates the requested probability value pertaining to the exponential distribution.
        If only one parameter is passed, then this function will return the value of the exponential CDF up to the value of that parameter: Pr(X <= x1).
        If two parameters are passed, then this function will return the value of the sum between those two values: Pr(x1 <= X <= x2).

        Parameters
        ----------
        lmbd: float
            The rate parameter of the exponential distribution. Takes on positive real values.
        x1: int
            The first threshold value for the probability.
        x2: int, optional
            The second threshold value for the probability.

        Raises
        ----------
        NotImplementedError
            If no parameters are passed in, if lmbd <= 0, or if x2 < x1.
        """

        if lmbd is None or x1 is None:
            raise NotImplementedError(
                "At least two parameters need to be passed into this function!"
            )

        elif lmbd <= 0:
            raise NotImplementedError("Invalid parameter!")

        elif x2 is not None and x2 < x1:
            raise NotImplementedError(
                "The third parameter should have a value of at least the value of the second parameter!"
            )

        if x2 is None:
            return 1 - expon.sf(x1, 0, 1 / lmbd)
        else:
            return expon.sf(x1, 0, 1 / lmbd) - expon.sf(x2, 0, 1 / lmbd)

    def calc_pr_beta(self):
        # TODO
        pass

    def calc_pr_gamma(self):
        # TODO
        pass

    def calc_pr_pareto(self):
        # TODO
        pass

    def calc_pr_norm(self, x1, x2=None, mu=0, sigma=1):
        """
        Calculates the requested probability value pertaining to the normal distribution.
        If only one parameter is passed, then this function will return the value of the normal CDF (phi) up to the value of that parameter: Pr(X <= x1).
        If two parameters are passed, then this function will return the value of the integral between those two values: Pr(x1 <= X <= x2).

        Parameters
        ----------
        x1: float
            The first value to get the value of the integral probability for from negative infinity to x1.
        x2: float, optional
            The second value, where the x2 >= x1 condition must be satisfied, to get the value of the integral's area between x1 and x2.
        mu: float, optional
            The mean of the normal distribution.
        sigma: float, optional
            The standard deviation of the normal distribution.

        Raises
        ----------
        NotImplementedError
            If no parameters are passed in or if x2 < x1.
        """

        if x1 is None:
            raise NotImplementedError(
                "At least one parameter needs to be passed into this function!"
            )

        elif x2 is not None and x2 < x1:
            raise NotImplementedError(
                "The second parameter should have a value of at least the value of the first parameter!"
            )

        if x2 is None:
            # Due to the symmetric nature
            if x1 <= mu:
                return norm.sf(-x1, mu, sigma)
            else:
                return 1 - norm.sf(x1, mu, sigma)
        else:
            return norm.sf(x1, mu, sigma) - norm.sf(x2, mu, sigma)

    def calc_pr_chi2(self, m, x1, x2=None):
        """
        Calculates the requested probability value pertaining to the central chi-squared distribution.
        If only one parameter is passed, then this function will return the value of the chi-squared CDF up to the value of that parameter: Pr(X <= x1).
        If two parameters are passed, then this function will return the value of the integral between those two values: Pr(x1 <= X <= x2).

        Parameters
        ----------
        m: int
            The degrees of freedom of the central chi-squared distribution.
        x1: float
            The first value to get the value of the integral probability for from negative infinity to x1.
        x2: float, optional
            The second value, where the x2 >= x1 condition must be satisfied, to get the value of the integral's area between x1 and x2.

        Raises
        ----------
        NotImplementedError
            If no required parameters are passed in, if m < 1 or if x2 < x1.
        """

        if m is None or x1 is None:
            raise NotImplementedError(
                "At least two parameters need to be passed into this function!"
            )

        elif m < 1:
            raise NotImplementedError("The degrees of freedom should be at least one!")

        elif x2 is not None and x2 < x1:
            raise NotImplementedError(
                "The third parameter should have a value of at least the value of the second parameter!"
            )

        if x2 is None:
            return 1 - chi2.sf(x1, m)
        else:
            return chi2.sf(x1, m) - chi2.sf(x2, m)

    def calc_pr_t(self, m, x1, x2=None):
        """
        Calculates the requested probability value pertaining to the central Student's t-distribution.
        If only one parameter is passed, then this function will return the value of the t-distribution CDF up to the value of that parameter: Pr(X <= x1).
        If two parameters are passed, then this function will return the value of the integral between those two values: Pr(x1 <= X <= x2).

        Parameters
        ----------
        m: int
            The degrees of freedom of the central Student's t-distribution.
        x1: float
            The first value to get the value of the integral probability for from negative infinity to x1.
        x2: float, optional
            The second value, where the x2 >= x1 condition must be satisfied, to get the value of the integral's area between x1 and x2.

        Raises
        ----------
        NotImplementedError
            If no required parameters are passed in, if m < 1 or if x2 < x1.
        """

        if m is None or x1 is None:
            raise NotImplementedError(
                "At least two parameters need to be passed into this function!"
            )

        elif m < 1:
            raise NotImplementedError("The degrees of freedom should be at least one!")

        elif x2 is not None and x2 < x1:
            raise NotImplementedError(
                "The third parameter should have a value of at least the value of the second parameter!"
            )

        if x2 is None:
            # Due to the symmetric nature
            if x1 <= 0:
                return t.sf(-x1, m)
            else:
                return 1 - t.sf(x1, m)
        else:
            return t.sf(x1, m) - t.sf(x2, m)

    def find_inv_bernoulli(self):
        # TODO
        pass

    def find_inv_binom(self):
        # TODO
        pass

    def find_inv_geom(self):
        # TODO
        pass

    def find_inv_poisson(self):
        # TODO
        pass

    def find_inv_expon(self):
        # TODO
        pass

    def find_inv_beta(self):
        # TODO
        pass

    def find_inv_gamma(self):
        # TODO
        pass

    def find_inv_pareto(self):
        # TODO
        pass

    def find_inv_norm(self, p, mode="regular", mu=0, sigma=1):
        """
        Calculates the value of x (or values of x1 and x2) at which Pr(X <= x) = p (regular), Pr(X > x) = p (reversed) or Pr(x1 <= X <= x2) = p (interval).

        Parameters
        ----------
        p: float
            The value of the desired probability. Takes on values where 0 <= p <= 1.
        mode: str
            The mode at which the calculation is performed. Can take on 3 values: "regular", "reversed" or "interval".
        mu: float, optional
            The mean of the normal distribution.
        sigma: float, optional
            The standard deviation of the normal distribution.

        Raises
        ----------
        NotImplementedError
            If no required parameters are passed in.
        """

        if p is None:
            raise NotImplementedError(
                "At least one parameter needs to be passed into this function!"
            )

        elif mode not in ["regular", "reversed", "interval"]:
            raise NotImplementedError("Invalid mode parameter!")

        elif p < 0 or p > 1:
            raise NotImplementedError("Invalid probabilistic constraint value!")

        if mode == "regular":
            return norm.ppf(p, mu, sigma)
        elif mode == "reversed":
            return norm.ppf(1 - p, mu, sigma)
        elif mode == "interval":
            return norm.interval(p, mu, sigma)

    def find_inv_chi2(self, m, p, mode="regular"):
        """
        Calculates the value of x (or values of x1 and x2) at which Pr(X <= x) = p (regular), Pr(X > x) = p (reversed) or Pr(x1 <= X <= x2) = p (interval).

        Parameters
        ----------
        m: float
            The degrees of freedom of the central chi-squared distribution.
        p: float
            The value of the desired probability. Takes on values where 0 <= p <= 1.
        mode: str
            The mode at which the calculation is performed. Can take on 3 values: "regular", "reversed" or "interval".

        Raises
        ----------
        NotImplementedError
            If no required parameters are passed in or if m < 1.
        """

        if m is None or p is None:
            raise NotImplementedError(
                "At least two parameters needs to be passed into this function!"
            )

        elif m < 1:
            raise NotImplementedError("The degrees of freedom should be at least one!")

        elif p < 0 or p > 1:
            raise NotImplementedError("Invalid probabilistic constraint value!")

        elif mode not in ["regular", "reversed", "interval"]:
            raise NotImplementedError("Invalid mode parameter!")

        if mode == "regular":
            return chi2.ppf(p, m)
        elif mode == "reversed":
            return chi2.ppf(1 - p, m)
        elif mode == "interval":
            return chi2.interval(p, m)

    def find_inv_t(self, m, p, mode="regular"):
        """
        Calculates the value of x (or values of x1 and x2) at which Pr(X <= x) = p (regular), Pr(X > x) = p (reversed) or Pr(x1 <= X <= x2) = p (interval).

        Parameters
        ----------
        m: float
            The degrees of freedom of the central Student's t-distribution.
        p: float
            The value of the desired probability. Takes on values where 0 <= p <= 1.
        mode: str
            The mode at which the calculation is performed. Can take on 3 values: "regular", "reversed" or "interval".

        Raises
        ----------
        NotImplementedError
            If no required parameters are passed in or if m < 1.
        """

        if m is None or p is None:
            raise NotImplementedError(
                "At least two parameters needs to be passed into this function!"
            )

        elif m < 1:
            raise NotImplementedError("The degrees of freedom should be at least one!")

        elif p < 0 or p > 1:
            raise NotImplementedError("Invalid probabilistic constraint value!")

        elif mode not in ["regular", "reversed", "interval"]:
            raise NotImplementedError("Invalid mode parameter!")

        if mode == "regular":
            return t.ppf(p, m)
        elif mode == "reversed":
            return t.ppf(1 - p, m)
        elif mode == "interval":
            return t.interval(p, m)

    def get_mle_bernoulli(self, random_sample):
        return np.average(random_sample)

    def get_mle_binom(self, N, random_sample):
        return np.average(random_sample) / N

    def get_mle_geom(self, random_sample):
        return 1 / (1 + np.average(random_sample))

    def get_mle_poisson(self, random_sample):
        return np.average(random_sample)

    def get_mle_expon(self, random_sample):
        return 1 / np.average(random_sample)

    def get_mle_gamma(self, random_sample):
        # TODO
        pass

    def get_mle_norm(self, random_sample):
        return (np.average(random_sample), np.std(random_sample))

    def get_posterior_beta_params_bernoulli_sampling(self, alpha, beta, random_sample):
        return (
            alpha + (np.array(random_sample) == 1).sum(),
            beta + (np.array(random_sample) == 0).sum(),
        )

    def get_posterior_beta_params_binom_sampling(self, N, alpha, beta, random_sample):
        return (
            alpha + np.sum(random_sample),
            beta + (N * len(random_sample)) - np.sum(random_sample),
        )

    def get_posterior_beta_params_geom_sampling(self):
        # TODO
        pass

    def get_posterior_gamma_params_poisson_sampling(self, alpha, beta, random_sample):
        return (alpha + np.sum(random_sample), beta + len(random_sample))

    def get_posterior_gamma_params_expon_sampling(self, alpha, beta, random_sample):
        return (alpha + len(random_sample), beta + np.sum(random_sample))

    def get_posterior_gamma_params_gamma_sampling(
        self, alpha, beta, alpha_not, random_sample
    ):
        return (alpha + (len(random_sample) * alpha_not), beta + np.sum(random_sample))

    def get_posterior_norm_params_norm_sampling(
        self, actual_sigma, prior_mu, prior_sigma, random_sample
    ):
        return (
            (
                (actual_sigma * actual_sigma * prior_mu)
                + (prior_sigma * prior_sigma * np.sum(random_sample))
            )
            / (
                (actual_sigma * actual_sigma)
                + (len(random_sample) * prior_sigma * prior_sigma)
            ),
            (actual_sigma * actual_sigma * prior_sigma * prior_sigma)
            / (
                (actual_sigma * actual_sigma)
                + (len(random_sample) * prior_sigma * prior_sigma)
            ),
        )

    def get_posterior_pareto_params_uniform_sampling(self, scale, shape, random_sample):
        return (np.amax(np.append(random_sample, scale)), shape + len(random_sample))

    def get_symmetric_r_value_in_terms_of_biased_sample_stddev_for_confidence_interval_for_mean(
        self, n, p
    ):
        """
        Solves for Pr((sample_mean - r) < actual_true_mean < (sample_mean + r)) = p in terms of biased sample stddev.
        """

        (left_r, right_r) = self.find_inv_t(n - 1, p, "interval")
        factor = n / (n - 1)

        return (factor * left_r / np.sqrt(n), factor * right_r / np.sqrt(n))

    def get_symmetric_r_value_in_terms_of_unbiased_sample_stddev_for_confidence_interval_for_mean(
        self, n, p
    ):
        """
        Solves for Pr((sample_mean - r) < actual_true_mean < (sample_mean + r)) = p in terms of unbiased sample stddev.
        """

        (left_r, right_r) = self.find_inv_t(n - 1, p, "interval")

        return (left_r / np.sqrt(n), right_r / np.sqrt(n))

    def get_observed_value_of_confidence_interval_of_mean(self, p, random_sample):
        (
            left_r,
            right_r,
        ) = self.get_symmetric_r_value_in_terms_of_unbiased_sample_stddev_for_confidence_interval_for_mean(
            len(random_sample), p
        )

        mu = np.average(random_sample)
        sigma = np.std(random_sample, ddof=1)

        return (mu + (left_r * sigma), mu + (right_r * sigma))

    def get_c_factors_in_terms_of_biased_sample_stddev_for_confidence_interval_for_variance(
        self, n, p
    ):
        """
        Solves for Pr((c1 * biased_sample_variance) < true_actual_variance < (c2 * biased_sample_variance)) = p in terms of biased sample stddev.
        """

        (left_c, right_c) = self.find_inv_chi2(n - 1, p, "interval")

        return (n / right_c, n / left_c)

    def get_c_factors_in_terms_of_unbiased_sample_stddev_for_confidence_interval_for_variance(
        self, n, p
    ):
        """
        Solves for Pr((c1 * biased_sample_variance) < true_actual_variance < (c2 * biased_sample_variance)) = p in terms of unbiased sample stddev.
        """

        (
            left_c,
            right_c,
        ) = self.get_c_factors_in_terms_of_biased_sample_stddev_for_confidence_interval_for_variance(
            n, p
        )
        factor = (n - 1) / n

        return (factor * left_c, factor * right_c)

    def get_observed_value_of_confidence_interval_of_variance(self, p, random_sample):
        (
            left_c,
            right_c,
        ) = self.get_c_factors_in_terms_of_biased_sample_stddev_for_confidence_interval_for_variance(
            len(random_sample), p
        )

        sigma = np.std(random_sample)

        return (left_c * sigma * sigma, right_c * sigma * sigma)

    def get_min_chi2_n(
        self,
        required_min_prob_value,
        smaller_factor,
        min_constraint=2,
        larger_factor=None,
    ):
        """
        Calculates the minimum n that satisfy a certain interval of probabilistic constraint for a central chi-squared distribution with n-1 degrees of freedom.
        The constraint would be of either format:
        - Pr(X <= (smaller_factor * n)) >= required_min_prob_value
        - Pr((smaller_factor * n) <= X <= (larger_factor * n)) >= required_min_prob_value

        Parameters
        ----------
        required_min_prob_value: float
            The required probabilistic constraint. Takes on values where 0 <= required_min_prob_value <= 1.
        smaller_factor: float
            The smaller multiplier factor of the constraint.
        min_constraint: float, optional
            The additional minimum constraint placed upon the value of n.
        larger_factor: float, optional
            The larger multiplier factor of the constraint, where larger_factor >= smaller_factor.

        Raises
        ----------
        NotImplementedError
            If no required parameters are passed in, if required_min_prob_value < 0, if required_min_prob_value > 1 or if larger_factor < smaller_factor.
        """

        if required_min_prob_value is None or smaller_factor is None:
            raise NotImplementedError(
                "At least two parameters need to be passed into this function!"
            )

        elif required_min_prob_value < 0 or required_min_prob_value > 1:
            raise NotImplementedError("Invalid probabilistic constraint value!")

        elif larger_factor is not None and larger_factor < smaller_factor:
            raise NotImplementedError(
                "The third parameter should have a value of at least the value of the second parameter!"
            )

        if larger_factor is None:
            temp_n = min_constraint

            while (
                1 - chi2.sf(smaller_factor * temp_n, temp_n - 1)
            ) < required_min_prob_value:
                temp_n += 1

            return temp_n
        else:
            temp_n = min_constraint
            while (
                chi2.sf(smaller_factor * temp_n, temp_n - 1)
                - chi2.sf(larger_factor * temp_n, temp_n - 1)
            ) < required_min_prob_value:
                temp_n += 1

            return temp_n

    def get_critical_value_norm(
        self, number_of_tails, significance_level, mu=0, sigma=1
    ):
        """
        Assumes that the rejection region is of the form [c, ∞). If two-tailed test, the test statistic is assumed to be of an absolute symmetric form.
        Returns the critical value at which the power of a corresponding hypothesis test is maximized.
        """
        if number_of_tails == 1:
            return norm.ppf((1 - significance_level), mu, sigma)
        elif number_of_tails == 2:
            return norm.ppf((2 - significance_level) / 2, mu, sigma)
        else:
            return None

    def get_critical_value_t(
        self, number_of_tails, significance_level, degrees_of_freedom
    ):
        """
        Assumes that the rejection region is of the form [c, ∞) (or (-∞, c] for the reversed/opposite one-tailed t-test). If two-tailed test, the test statistic is assumed to be of an absolute symmetric form.
        Returns the critical value at which the power of a corresponding hypothesis test (t-test) is maximized.
        """
        if number_of_tails == 1:
            return t.ppf((1 - significance_level), degrees_of_freedom)
        elif number_of_tails == 2:
            return t.ppf((2 - significance_level) / 2, degrees_of_freedom)
        else:
            return None

    def get_critical_value_chi2(
        self, number_of_tails, significance_level, degrees_of_freedom
    ):
        """
        Assumes that the rejection region is of the form [c, ∞). If two-tailed test, the test statistic is assumed to be of an absolute symmetric form.
        Returns the critical value at which the power of a corresponding hypothesis test is maximized.
        """
        if number_of_tails == 1:
            return chi2.ppf((1 - significance_level), degrees_of_freedom)
        elif number_of_tails == 2:
            return chi2.ppf((2 - significance_level) / 2, degrees_of_freedom)
        else:
            return None

    def get_symmetric_rejection_region_norm(self, significance_level, mu=0, sigma=1):
        percentile = norm.ppf(1 - (significance_level / 2), mu, sigma)
        return [-percentile, percentile]

    def compare_and_check_whether_to_reject_null_hypothesis_norm(
        self,
        number_of_tails,
        significance_level,
        observed_test_statistic,
        mu=0,
        sigma=1,
    ):
        """
        Returns True if null hypothesis is rejected, and False if null hypothesis is not rejected.
        """
        c = self.get_critical_value_norm(number_of_tails, significance_level, mu, sigma)

        return observed_test_statistic >= c

    def compare_and_check_whether_to_reject_null_hypothesis_t(
        self, sample, mu, significance_level
    ):
        """
        Returns True if null hypothesis is rejected, and False if null hypothesis is not rejected.
        """
        # Should return the same value as ttest_1samp(sample, mu)[0] >= self.get_critical_value_t(2, significance_level, len(sample) - 1)
        return ttest_1samp(sample, mu)[1] < significance_level

    def find_p_value_t_test(
        self, number_of_tails, degrees_of_freedom, test_statistic_value
    ):
        """
        Degrees of freedom are either (n - 1) or (n + m - 2), depending on whether it is a one-sample or a two-sample t-test respectively.
        """

        if number_of_tails == 1:
            return 1 - self.calc_pr_t(degrees_of_freedom, test_statistic_value)
        elif number_of_tails == 2:
            return 2 * (1 - self.calc_pr_t(degrees_of_freedom, test_statistic_value))

    def calculate_minimum_total_error_poisson_hypothesis_test(
        self, lambda_nought, lambda_one, n
    ):
        """
        Assumes that Type I and Type II errors are equally weighted/important, lambda_one > lambda_nought > 0 and that the random sample is independent and identically distributed.
        Simple null hypothesis, H_0: lambda = lambda_nought.
        Simple alternative hypothesis, H_1: lambda = lambda_one.
        """

        if n < 0:
            raise NotImplementedError(
                "The random sample's size should be non-negative."
            )
        elif lambda_one <= lambda_nought:
            raise NotImplementedError(
                "The lambda for the alternative hypothesis should be larger than the lambda for the null hypothesis."
            )

        y = (n * (lambda_one - lambda_nought)) / np.log(lambda_one / lambda_nought)
        alpha = poisson.sf(y, n * lambda_nought)
        beta = 1 - poisson.sf(y, n * lambda_one)
        return alpha + beta

    def calculate_ump_critical_value_bernoulli_rvs(
        self, significance_level, theta_naught, n
    ):
        """
        Assumes that the null hypothesis, H_0, is of the form: theta >= theta_naught.
        Also assumes that the parameter space of theta is the interval [0, 1] and the rejection region is of the form (-∞, c].
        Utilizes the monotone likelihood ratio property of the joint pmf of the test statistic of the sum of the iid Bernoulli random variables.
        """

        for c in range(n + 1):
            if 1 - binom.sf(c, n, theta_naught) <= significance_level:
                continue
            else:
                if (c - 1) < 0:
                    raise NotImplementedError("No valid values of c are found!")
                return c - 1

        raise NotImplementedError("No valid values of c are found!")

    def power_of_most_powerful_hypothesis_test_single_observation_of_unknown_dist_uniform_vs_standard_normal(
        self, significance_level, theta
    ):
        """
        Returns the power of the most powerful test when the null hypothesis is rejected.

        Hypothesis Test:
        - Null hypothesis: Unknown distribution is the uniform distribution on the interval [0, theta].
        - Alternative hypothesis: Unknown distribution is the standard normal distribution.
        """

        if theta < significance_level:
            raise NotImplementedError(
                "Invalid inputs that do not make sense! Please check your inputs again."
            )

        c = uniform.ppf(significance_level, 0, 0 + theta)

        return (1 - norm.sf(c)) + norm.sf(theta)

    def get_critical_value_and_p_value_chi_squared_test(self, sample, probability_list):
        """
        Returns the critical value and the p-value of the specified collection of chi-squared tests in that order in a tuple.
        Assumes that the rejection region is of the form [c, ∞).
        """

        if len(sample) <= 0 or len(probability_list) <= 0:
            raise NotImplementedError(
                "Invalid lengths of either the sample list or the probability list are entered!"
            )
        elif len(sample) != len(probability_list):
            raise NotImplementedError(
                "The length of the sample array should correspond to the length of the probability array."
            )
        elif np.sum(probability_list) != 1:
            raise NotImplementedError(
                "The total probability in the probability list should be equal to 1 for the test to make sense!"
            )

        expected_values_array = np.multiply(np.sum(sample), probability_list)
        output = chisquare(sample, f_exp=expected_values_array)
        return (output[0], output[1])

    def get_best_fit_polynomial_curve_with_least_squares_method(
        self, x_data, y_data, n_limit, desired_max_error_threshold=np.inf, desired_max_error_difference_threshold=np.inf
    ):
        """
        Assumes constant zero noise. Due to the Stone–Weierstrass theorem, we only obtain and return the best least-squares fit polynomial curve with the most meaningful reduction in the least-squares error, if such a curve exists.
        An optional maximum error threshold parameter exists to ensure that the residual is less than or equal to the specified threshold.
        Another optional maximum error DIFFERENCE threshold parameter also exists to ensure that the current (or next, in this context) error difference is less than or equal to the specified threshold.
        """

        previous_error = np.inf
        previous_error_term_difference = np.inf
        current_poly_degree = 0
        output_coeffs = None

        while current_poly_degree <= n_limit:
            result = poly.polyfit(x_data, y_data, current_poly_degree, full=True)
            coeffs = result[0]  # Get coefficients
            current_error = result[1][0][0]  # Get total residuals
            if current_poly_degree >= 2:
                current_error_term_difference = previous_error - current_error
                if (
                    current_error_term_difference <= previous_error_term_difference
                    and previous_error <= desired_max_error_threshold
                    and current_error_term_difference <= desired_max_error_difference_threshold
                ):
                    break
                previous_error_term_difference = current_error_term_difference
            output_coeffs = coeffs
            previous_error = current_error
            current_poly_degree += 1

        # Format and pretty-print the polynomial equation output
        output = "Polynomial equation: y ="
        for idx, coeff in enumerate(output_coeffs):
            if idx == 0:
                output += f" {coeff:.2f}"
            else:
                if coeff < 0:
                    output += f" - {-coeff:.2f}x^{idx}"
                else:
                    output += f" + {coeff:.2f}x^{idx}"

        return output

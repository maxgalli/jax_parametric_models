"""Extended negative log-likelihood for statistical fitting."""

from __future__ import annotations

import evermore as evm
import jax.numpy as jnp
from jaxtyping import Array, Float

from .distributions import SumPDF

__all__ = ["create_extended_nll"]


def create_extended_nll(
    params, sum_pdf: SumPDF, data: Float[Array, "N"]
) -> Float[Array, ""]:
    """Create extended negative log-likelihood.

    The extended likelihood includes:
    1. Poisson term for the total event count: P(N|nu) = exp(-nu) * nu^N / N!
    2. Sum of log-probabilities for individual events
    3. Prior constraints on parameters

    Args:
        params: PyTree of evermore Parameters (for computing priors)
        sum_pdf: SumPDF instance containing PDFs and their expected counts
        data: Observed data array

    Returns:
        Negative log-likelihood value (scalar)
    """
    # Get total expected count from SumPDF
    extended_vals = (
        sum_pdf.extended_vals.value
        if hasattr(sum_pdf.extended_vals, "value")
        else sum_pdf.extended_vals
    )
    nu_total = sum(extended_vals)

    # Number of observed events
    N = len(data)

    # Poisson term: P(N|nu) = exp(-nu) * nu^N / N!
    # log P(N|nu) = -nu + N*log(nu) - log(N!)
    # We drop log(N!) since it's constant
    poisson_term = -nu_total + N * jnp.log(nu_total)

    # Get weighted sum of normalized probabilities
    sum_probs = sum_pdf.prob(data)

    # Sum of log-likelihoods (add small constant for numerical stability)
    log_probs = jnp.log(sum_probs + 1e-8)
    log_likelihood = poisson_term + jnp.sum(log_probs)

    # Add priors using evm.loss.get_log_probs
    constraints = evm.loss.get_log_probs(params)
    prior_values = [v for v in constraints.values()]
    if prior_values:
        prior_total = jnp.sum(jnp.array(prior_values))
        log_likelihood += prior_total

    # Return negative log-likelihood
    return -log_likelihood


def create_nll(params, sum_pdf: SumPDF, data: Float[Array, "N"]) -> Float[Array, ""]:
    """Create negative log-likelihood.

    The likelihood includes:
    1. Sum of log-probabilities for individual events
    2. Prior constraints on parameters

    Args:
        params: PyTree of evermore Parameters (for computing priors)
        sum_pdf: SumPDF instance containing PDFs and their expected counts
        data: Observed data array

    Returns:
        Negative log-likelihood value (scalar)
    """
    # Get weighted sum of normalized probabilities
    sum_probs = sum_pdf.prob(data)

    # Sum of log-likelihoods (add small constant for numerical stability)
    log_probs = jnp.log(sum_probs + 1e-8)
    log_likelihood = jnp.sum(log_probs)

    # Add priors using evm.loss.get_log_probs
    constraints = evm.loss.get_log_probs(params)
    prior_values = [v for v in constraints.values()]
    if prior_values:
        prior_total = jnp.sum(jnp.array(prior_values))
        log_likelihood += prior_total

    # Return negative log-likelihood
    return -log_likelihood

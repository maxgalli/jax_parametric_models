"""Probability distributions and parameterized functions for statistical modeling."""

from __future__ import annotations

import abc

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float
from quadax import quadgk

# ============================================================================
# BasePDF classes
# ============================================================================


class BasePDF(nnx.Pytree):
    """Abstract base class for PDFs storing VALUES (JAX arrays)."""

    def unnormalized_prob(self, x: Float[Array, "..."]) -> Float[Array, "..."]:
        """Return unnormalized probability density at x."""
        raise NotImplementedError

    def integrate(self, lower=None, upper=None) -> Float[Array, ""]:
        """Integrate the unnormalized probability over [lower, upper]."""
        lower = self.lower if lower is None else lower
        upper = self.upper if upper is None else upper
        epsabs = epsrel = 1e-5
        integral, _ = quadgk(
            self.unnormalized_prob, [lower, upper], epsabs=epsabs, epsrel=epsrel
        )
        return integral

    def prob(self, x: Float[Array, "..."]) -> Float[Array, "..."]:
        """Return normalized probability density at x."""
        norm = self.integrate()
        return self.unnormalized_prob(x) / norm

    def log_prob(self, x: Float[Array, "..."]) -> Float[Array, "..."]:
        """Return log of normalized probability density at x."""
        return jnp.log(self.prob(x))

    def sample(self, key, n_events: int) -> Float[Array, "n_events"]:
        """Sample n_events from the distribution.

        Args:
            key: JAX random key
            n_events: Number of events to sample

        Returns:
            Array of sampled values
        """
        raise NotImplementedError


class Gaussian(BasePDF):
    """Gaussian PDF with fixed mean and standard deviation."""

    def __init__(
        self,
        mu: Float[Array, ""] | float,
        sigma: Float[Array, ""] | float,
        lower: Float[Array, ""] | float,
        upper: Float[Array, ""] | float,
    ):
        """Initialize Gaussian PDF.

        Args:
            mu: Mean of the Gaussian
            sigma: Standard deviation of the Gaussian
            lower: Lower bound for the observable
            upper: Upper bound for the observable
        """
        self.mu = jnp.asarray(mu)
        self.sigma = jnp.asarray(sigma)
        self.lower = jnp.asarray(lower)
        self.upper = jnp.asarray(upper)

    def unnormalized_prob(self, x: Float[Array, "..."]) -> Float[Array, "..."]:
        return jnp.exp(-0.5 * ((x - self.mu) / self.sigma) ** 2)

    def sample(self, key, n_events: int) -> Float[Array, "n_events"]:
        return jax.random.normal(key, shape=(n_events,)) * self.sigma + self.mu


class Exponential(BasePDF):
    """Exponential PDF with fixed rate parameter."""

    def __init__(
        self,
        lambd: Float[Array, ""] | float,
        lower: Float[Array, ""] | float,
        upper: Float[Array, ""] | float,
    ):
        """Initialize Exponential PDF.

        Args:
            lambd: Rate parameter (inverse of mean)
            lower: Lower bound for the observable
            upper: Upper bound for the observable
        """
        self.lambd = jnp.asarray(lambd)
        self.lower = jnp.asarray(lower)
        self.upper = jnp.asarray(upper)

    def unnormalized_prob(self, x: Float[Array, "..."]) -> Float[Array, "..."]:
        return jnp.exp(-self.lambd * x)

    def sample(self, key, n_events: int) -> Float[Array, "n_events"]:
        """Sample from truncated exponential using inverse CDF."""
        u = jax.random.uniform(key, shape=(n_events,))
        z = jnp.exp(-self.lambd * self.lower) - u * (
            jnp.exp(-self.lambd * self.lower) - jnp.exp(-self.lambd * self.upper)
        )
        return -jnp.log(z) / self.lambd


class SumPDF(BasePDF):
    """Weighted sum of multiple PDFs.

    This class represents a mixture of PDFs weighted by their expected counts.
    The prob() method returns the weighted average of the component PDFs'
    normalized probabilities.

    Args:
        pdfs: List of PDF instances
        extended_vals: List of expected event counts (one per PDF)
        lower: Lower bound for the observable
        upper: Upper bound for the observable
    """

    def __init__(
        self,
        pdfs: list[BasePDF],
        extended_vals: list[Float[Array, ""] | float],
        lower: Float[Array, ""] | float,
        upper: Float[Array, ""] | float,
    ):
        """Initialize SumPDF.

        Args:
            pdfs: List of PDF instances
            extended_vals: List of expected event counts (floats or arrays)
            lower: Lower bound for the observable
            upper: Upper bound for the observable
        """
        self.pdfs = nnx.data(pdfs)
        self.extended_vals = nnx.data(extended_vals)
        self.lower = jnp.asarray(lower)
        self.upper = jnp.asarray(upper)

    def prob(self, x: Float[Array, "..."]) -> Float[Array, "..."]:
        """Return weighted sum of normalized probabilities.

        Computes: sum_i (nu_i / nu_total) * p_i(x)
        where p_i(x) is the normalized probability from PDF i.
        """
        # Unwrap data if needed
        pdfs = self.pdfs.value if hasattr(self.pdfs, "value") else self.pdfs
        extended_vals = (
            self.extended_vals.value
            if hasattr(self.extended_vals, "value")
            else self.extended_vals
        )

        # Total expected count
        nu_total = sum(extended_vals)

        # Compute weighted sum of normalized probabilities
        result = jnp.zeros_like(x)
        for pdf, extended_val in zip(pdfs, extended_vals):
            weight = extended_val / nu_total
            result = result + weight * pdf.prob(x)

        return result

    def unnormalized_prob(self, x: Float[Array, "..."]) -> Float[Array, "..."]:
        """Not used for SumPDF, raises NotImplementedError."""
        raise NotImplementedError(
            "SumPDF computes weighted sum of normalized PDFs, use prob() instead"
        )

    def log_prob(self, x: Float[Array, "..."]) -> Float[Array, "..."]:
        """Return log of weighted sum probability."""
        return jnp.log(self.prob(x))

    def sample(self, key, n_events: int) -> Float[Array, "..."]:
        """Sample from mixture distribution (fixed event count).

        Uses exact expected counts (no Poisson fluctuation).
        For extended likelihood toys, use sample_extended() instead.

        Args:
            key: JAX random key
            n_events: Total number of events to sample (ignored, uses expected counts)

        Returns:
            Array of sampled events
        """
        # Unwrap data if needed
        pdfs = self.pdfs.value if hasattr(self.pdfs, "value") else self.pdfs
        extended_vals = (
            self.extended_vals.value
            if hasattr(self.extended_vals, "value")
            else self.extended_vals
        )

        # For each component, sample according to its expected count
        samples = []
        for pdf, extended_val in zip(pdfs, extended_vals):
            key, subkey = jax.random.split(key)
            # Number of events from this component
            n_component = int(jnp.round(extended_val))
            if n_component > 0:
                component_samples = pdf.sample(subkey, n_events=n_component)
                samples.append(component_samples)

        if samples:
            return jnp.concatenate(samples)
        else:
            return jnp.array([])

    def sample_extended(self, key) -> Float[Array, "..."]:
        """Sample from extended mixture distribution with Poisson fluctuation.

        For each component:
        1. Poisson-sample the number of events from expected count
        2. Sample that many events from the component PDF
        3. Concatenate all samples

        This is the correct way to generate toys for extended likelihood fits.

        Args:
            key: JAX random key

        Returns:
            Array of sampled events (variable length due to Poisson)
        """
        # Unwrap data if needed
        pdfs = self.pdfs.value if hasattr(self.pdfs, "value") else self.pdfs
        extended_vals = (
            self.extended_vals.value
            if hasattr(self.extended_vals, "value")
            else self.extended_vals
        )

        samples = []
        for pdf, extended_val in zip(pdfs, extended_vals):
            # Poisson-sample the number of events
            key, subkey = jax.random.split(key)
            n_events = jax.random.poisson(lam=extended_val, key=subkey, shape=())
            n_events = jnp.asarray(n_events, dtype=jnp.int32)

            # Sample events from this component PDF
            key, subkey = jax.random.split(key)
            if n_events > 0:
                component_samples = pdf.sample(subkey, int(n_events))
                samples.append(component_samples)

        if samples:
            return jnp.concatenate(samples)
        else:
            return jnp.array([])

    def sample_extended_fixed(
        self, key, max_events: int
    ) -> tuple[Float[Array, "max_events"], Float[Array, "max_events"]]:
        """Sample from extended mixture with fixed output size (vmap-friendly).

        Samples exactly max_events from the mixture, then uses Poisson fluctuation
        to determine how many are "real" events vs padding.

        Args:
            key: JAX random key
            max_events: Maximum number of events to sample (fixed size)

        Returns:
            samples: (max_events,) array of sampled values
            mask: (max_events,) boolean array, True for valid events based on Poisson draw
        """
        # Unwrap data if needed
        pdfs = self.pdfs.value if hasattr(self.pdfs, "value") else self.pdfs
        extended_vals = (
            self.extended_vals.value
            if hasattr(self.extended_vals, "value")
            else self.extended_vals
        )

        # Split keys for Poisson draws and sampling
        key_poisson, key_sample = jax.random.split(key)

        # Poisson-sample total number of events from sum of expectations
        nu_total = sum(extended_vals)
        n_total = jax.random.poisson(key_poisson, lam=nu_total, shape=())

        # Sample max_events from the mixture (using mixture weights)
        # Component probabilities (normalized by total expected count)
        weights = jnp.array([ext_val / nu_total for ext_val in extended_vals])

        # For each of max_events samples, choose which component it comes from
        key_component, key_values = jax.random.split(key_sample)
        component_keys = jax.random.split(key_values, max_events)

        # Sample component indices using categorical distribution
        component_indices = jax.random.categorical(
            key_component, logits=jnp.log(weights), shape=(max_events,)
        )

        # Sample one event from each component (will select based on component_indices)
        def sample_from_component(k, comp_idx):
            """Sample from the component indicated by comp_idx."""
            # Sample from each PDF
            samples_per_pdf = jnp.array([pdf.sample(k, 1)[0] for pdf in pdfs])
            # Select the one corresponding to comp_idx
            return samples_per_pdf[comp_idx]

        samples = jax.vmap(sample_from_component)(component_keys, component_indices)

        # Create mask: first n_total events are valid
        idx = jnp.arange(max_events)
        mask = idx < n_total

        return samples, mask

from __future__ import annotations

import abc
from typing import Optional, Sequence, Set, Tuple

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float
from quadax import quadgk

import evermore as evm


class ParameterizedFunction(nnx.Pytree):
    """Base class for functions that depend on evermore Parameters.

    Subclasses should store evm.Parameter objects as attributes and implement
    a .value property that returns the computed result as a jnp.array.
    """

    @property
    @abc.abstractmethod
    def value(self) -> Float[Array, "..."]:
        """Compute and return the function value."""
        raise NotImplementedError


class Distribution(nnx.Pytree):
    """Base helper around ``evermore`` parameters to describe PDFs."""

    def __init__(
        self,
        *,
        var: evm.Parameter,
        extended: Optional[evm.Parameter] = None,
    ) -> None:
        if extended is None:
            extended = evm.Parameter(
                value=jnp.array(1.0, dtype=var.value.dtype),
                name=f"{var.name}_extended",
                lower=jnp.array(0.0, dtype=var.value.dtype),
                upper=None,
                frozen=False,
            )
        if not isinstance(extended, (evm.Parameter, ParameterizedFunction)):
            raise TypeError(
                "extended must be an evermore.Parameter, ParameterizedFunction, or None; "
                f"got {type(extended)}"
            )

        self.var = var
        self.extended = extended

    @abc.abstractmethod
    def unnormalized_prob(self, x: Float[Array, "..."]) -> Float[Array, "..."]:
        raise NotImplementedError

    @abc.abstractmethod
    def sample(self, sample_shape, seed=None, **kwargs) -> Float[Array, "..."]:
        raise NotImplementedError

    def prob(self, x: Float[Array, "..."]) -> Float[Array, "..."]:
        norm = self.integrate()
        return self.unnormalized_prob(x) / norm

    def log_prob(self, x: Float[Array, "..."]) -> Float[Array, "..."]:
        return jnp.log(self.prob(x))

    def integrate(self, lower=None, upper=None) -> Float[Array, "..."]:
        lower = self.var.lower if lower is None else lower
        upper = self.var.upper if upper is None else upper

        epsabs = epsrel = 1e-5
        integral, _ = quadgk(
            self.unnormalized_prob,
            [lower, upper],
            epsabs=epsabs,
            epsrel=epsrel,
        )
        return integral

    def __call__(self, x: Float[Array, "..."]) -> Float[Array, "..."]:
        return self.prob(x)


class Gaussian(Distribution):
    """Gaussian distribution with mean and standard deviation as parameters."""

    def __init__(
        self,
        *,
        var: evm.Parameter,
        mu: evm.Parameter,
        sigma: evm.Parameter,
        extended: Optional[evm.Parameter] = None,
    ) -> None:
        super().__init__(var=var, extended=extended)
        self.mu = mu
        self.sigma = sigma

    def unnormalized_prob(self, value):
        return jnp.exp(-0.5 * ((value - self.mu.value) / self.sigma.value) ** 2)

    def sample(self, sample_shape, seed=None, **kwargs):
        return (
            jax.random.normal(seed, shape=sample_shape, dtype=self.mu.value.dtype)
            * self.sigma.value
            + self.mu.value
        )


class Exponential(Distribution):
    def __init__(
        self,
        *,
        var: evm.Parameter,
        lambd: evm.Parameter,
        extended: Optional[evm.Parameter] = None,
    ) -> None:
        super().__init__(var=var, extended=extended)
        self.lambd = lambd

    def unnormalized_prob(self, value):
        return jnp.exp(-self.lambd.value * value)

    def sample(self, sample_shape, seed=None, **kwargs):
        lower = self.var.lower
        upper = self.var.upper
        u = jax.random.uniform(seed, shape=sample_shape)
        lambda_val = self.lambd.value
        z = jnp.exp(-lambda_val * lower) - u * (
            jnp.exp(-lambda_val * lower) - jnp.exp(-lambda_val * upper)
        )
        return -jnp.log(z) / lambda_val


class SumExtended(ParameterizedFunction):
    """Compute the sum of extended values from multiple PDFs."""

    def __init__(self, pdfs: Tuple[Distribution, ...]):
        self._pdfs = nnx.data(pdfs)

    @property
    def pdfs(self) -> Tuple[Distribution, ...]:
        data = self._pdfs
        if hasattr(data, "value"):
            return data.value  # type: ignore[return-value]
        return data

    @property
    def value(self):
        totals = jnp.stack([jnp.asarray(pdf.extended.value) for pdf in self.pdfs])
        return jnp.sum(totals, axis=0)


class SumPDF(Distribution):
    def __init__(
        self,
        *,
        var: evm.Parameter,
        pdfs: Sequence[Distribution],
    ) -> None:
        pdfs_tuple = tuple(pdfs)
        extended = SumExtended(pdfs_tuple)
        super().__init__(var=var, extended=extended)
        self._pdfs = nnx.data(pdfs_tuple)

    @property
    def pdfs(self) -> Tuple[Distribution, ...]:
        data = self._pdfs
        if hasattr(data, "value"):
            return data.value  # type: ignore[return-value]
        return data

    def unnormalized_prob(self, x: Float[Array, "..."]) -> Float[Array, "..."]:
        weights = jnp.stack(
            [
                jnp.asarray(pdf.extended.value)
                / jnp.asarray(self.extended.value)
                for pdf in self.pdfs
            ],
            axis=0,
        )
        stacked = jnp.stack([pdf(x) for pdf in self.pdfs], axis=0)
        reshape_dims = (weights.shape[0],) + (1,) * (stacked.ndim - 1)
        weights = weights.reshape(reshape_dims)
        return jnp.sum(weights * stacked, axis=0)

    def prob(self, x: Float[Array, "..."]) -> Float[Array, "..."]:
        return self.unnormalized_prob(x)

    def sample(self, key):
        samples = []
        for pdf in self.pdfs:
            n2 = jax.random.poisson(lam=pdf.extended.value, key=key, shape=())
            n2 = jnp.asarray(n2, dtype=jnp.int64)
            sample = pdf.sample(sample_shape=(n2,), seed=key)
            samples.append(sample)
        return jnp.concatenate(samples, axis=-1)


class ExtendedNLL:
    """Generalised negative log-likelihood for a sum of PDFs."""

    def __init__(self, model: Distribution) -> None:
        self.model = model
        # Collect all parameters with priors at initialization (before JIT)
        self._params_with_priors = self._collect_parameters_with_priors(model)

    def _collect_parameters_with_priors(self, node, found=None):
        """Recursively collect all parameters that have priors."""
        if found is None:
            found = []

        if isinstance(node, evm.Parameter):
            if getattr(node, "prior", None) is not None:
                # Check if not already in list (by identity)
                if all(p is not node for p in found):
                    found.append(node)
        else:
            # Traverse any object that might contain parameters
            for attr_name in dir(node):
                if attr_name.startswith('_') or attr_name in ('mro', 'value'):
                    continue
                try:
                    attr = getattr(node, attr_name)
                    # Check for evm.Parameter first, even if callable
                    if isinstance(attr, evm.Parameter):
                        self._collect_parameters_with_priors(attr, found)
                    elif not callable(attr) and (hasattr(attr, '__dict__') or isinstance(attr, (tuple, list))):
                        # Traverse objects that might contain parameters
                        if isinstance(attr, (tuple, list)):
                            for item in attr:
                                self._collect_parameters_with_priors(item, found)
                        elif not isinstance(attr, (str, int, float, bool, type(None))):
                            self._collect_parameters_with_priors(attr, found)
                except Exception:
                    pass
        return found

    def __call__(self, x):
        N = x.shape[0]
        nu_total = self.model.extended.value
        pdf = self.model.prob(x)

        poisson_term = -nu_total + N * jnp.log(nu_total)
        log_likelihood = poisson_term + jnp.sum(jnp.log(pdf + 1e-8))

        # Add priors for all parameters collected at initialization
        prior_total = jnp.array(0.0)
        for param in self._params_with_priors:
            prior_val = param.prior.log_prob(param.value)
            prior_total = prior_total + jnp.sum(prior_val)

        log_likelihood += prior_total
        return jnp.squeeze(-log_likelihood)


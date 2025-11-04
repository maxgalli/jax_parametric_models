from __future__ import annotations

import abc
from typing import Optional, Sequence, Set, Tuple

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float
from quadax import quadgk

import evermore as evm

from jax.tree_util import tree_leaves


class EVMDistribution(nnx.Module):
    """Base helper around ``evermore`` parameters to describe PDFs."""

    def __init__(
        self,
        *,
        var: evm.Parameter,
        extended: Optional[evm.Parameter] = None,
        modifier_parameters: Optional[Tuple[evm.Parameter, ...]] = None,
    ) -> None:
        if extended is None:
            extended = evm.Parameter(
                value=jnp.array(1.0, dtype=var.value.dtype),
                name=f"{var.name}_extended",
                lower=jnp.array(0.0, dtype=var.value.dtype),
                upper=None,
                frozen=False,
            )
        if not isinstance(extended, evm.Parameter):
            raise TypeError(
                "extended must be an evermore.Parameter or None; "
                f"got {type(extended)}"
            )

        self.var = var
        self.extended = extended
        self._modifier_parameters = nnx.data(tuple(modifier_parameters or ()))

    @property
    def modifier_parameters(self) -> Tuple[evm.Parameter, ...]:
        data = getattr(self, "_modifier_parameters", ())
        if hasattr(data, "value"):
            return data.value  # type: ignore[return-value]
        if isinstance(data, tuple):
            return data
        return tuple()

    @modifier_parameters.setter
    def modifier_parameters(self, value: Tuple[evm.Parameter, ...]) -> None:
        self._modifier_parameters = nnx.data(tuple(value))

    @property
    def pdfs(self) -> Tuple[EVMDistribution, ...]:
        data = getattr(self, "_pdfs", ())
        if hasattr(data, "value"):
            return data.value  # type: ignore[return-value]
        if isinstance(data, tuple):
            return data
        return tuple()

    @pdfs.setter
    def pdfs(self, value: Sequence[EVMDistribution]) -> None:
        self._pdfs = nnx.data(tuple(value))

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


class EVMGaussian(EVMDistribution):
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


class EVMExponential(EVMDistribution):
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


class EVMSumPDF(EVMDistribution):
    def __init__(
        self,
        *,
        var: evm.Parameter,
        pdfs: Sequence[EVMDistribution],
    ) -> None:
        totals = jnp.stack([jnp.asarray(pdf.extended.value) for pdf in pdfs])
        total = jnp.sum(totals, axis=0)
        extended = evm.Parameter(
            value=jnp.asarray(total, dtype=var.value.dtype),
            name=f"{var.name}_sum_extended",
            lower=jnp.array(0.0, dtype=var.value.dtype),
            upper=None,
            frozen=True,
        )
        super().__init__(var=var, extended=extended)
        self._pdfs = nnx.data(tuple(pdfs))

        modifier_params = list(self.modifier_parameters)
        seen = list(modifier_params)
        for pdf in self.pdfs:
            for param in pdf.modifier_parameters:
                if all(existing is not param for existing in seen):
                    seen.append(param)
                    modifier_params.append(param)
        self.modifier_parameters = tuple(modifier_params)

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

    def __init__(self, model: EVMDistribution) -> None:
        self.model = model

    def __call__(self, x):
        N = x.shape[0]
        nu_total = self.model.extended.value
        pdf = self.model.prob(x)

        poisson_term = -nu_total + N * jnp.log(nu_total)
        log_likelihood = poisson_term + jnp.sum(jnp.log(pdf + 1e-8))

        def _collect_prior_logprob(node, total, seen: Set[int]):
            if isinstance(node, evm.Parameter):
                node_id = id(node)
                if node_id not in seen and getattr(node, "prior", None) is not None:
                    seen.add(node_id)
                    prior_val = node.prior.log_prob(node.value)
                    total = total + jnp.sum(prior_val)
            return total, seen

        total = jnp.array(0.0)
        seen: Set[int] = set()
        for leaf in tree_leaves(self.model):
            total, seen = _collect_prior_logprob(leaf, total, seen)

        if hasattr(self.model, "modifier_parameters"):
            for param in self.model.modifier_parameters:
                if param is not None and getattr(param, "prior", None) is not None:
                    total = total + jnp.sum(param.prior.log_prob(param.value))

        log_likelihood += total
        return jnp.squeeze(-log_likelihood)


class GaussianConstraint:
    def __init__(self, param, mu, sigma):
        self.param = param
        self.mu = mu
        self.sigma = sigma

    def __call__(self):
        return -0.5 * ((self.param.value - self.mu) / self.sigma) ** 2

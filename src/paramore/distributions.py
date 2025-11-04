import jax
import jax.numpy as jnp
from jax import jit
from jax import lax
from functools import partial
import evermore as evm
import equinox as eqx
from quadax import romberg, quadgk

import equinox as eqx
import abc
from jaxtyping import Array, Float
from typing import Optional, Set, Tuple
from jax.tree_util import tree_leaves


class EVMDistribution(eqx.Module):
    var: evm.Parameter
    extended: Optional[evm.Parameter] = None
    modifier_parameters: Tuple[evm.Parameter, ...] = eqx.field(
        default_factory=tuple
    )

    def __post_init__(self):
        if self.extended is None:
            object.__setattr__(
                self,
                "extended",
                evm.Parameter(
                    value=1.0,
                    name=f"{self.var.name}_extended",
                    lower=0.0,
                    upper=None,
                    frozen=False,
                ),
            )
        elif not isinstance(self.extended, evm.Parameter):
            raise TypeError(
                "extended must be an evermore Parameter or None; "
                f"got {type(self.extended)}"
            )

    @abc.abstractmethod
    def unnormalized_prob(self, x: Float[Array, "..."]) -> Float[Array, "..."]:
        """Compute the unnormalized probability density function."""
        raise NotImplementedError("Subclasses must implement unnormalized_prob.")

    @abc.abstractmethod
    def sample(self, sample_shape, seed=None, **kwargs) -> Float[Array, "..."]:
        """Sample from the distribution."""
        raise NotImplementedError("Subclasses must implement sample.")

    def prob(self, x: Float[Array, "..."]) -> Float[Array, "..."]:
        norm = self.integrate()
        #return self.unnormalized_prob(x) * self.extended / norm
        return self.unnormalized_prob(x) / norm

    def log_prob(self, x: Float[Array, "..."]) -> Float[Array, "..."]:
        return jnp.log(self.prob(x))

    def integrate(self, lower=None, upper=None) -> Float[Array, "..."]:
        """Integrate the unnormalized probability density function over the range [lower, upper]."""
        if lower is None:
            lower = self.var.lower
        if upper is None:
            upper = self.var.upper

        #def _unnormalized_prob_nograd(value):
        #    return lax.stop_gradient(self.unnormalized_prob(value))

        epsabs = epsrel = 1e-5
        #integral, _ = romberg(
        integral, _ = quadgk(
            self.unnormalized_prob,
            #_unnormalized_prob_nograd,
            [lower, upper],
            epsabs=epsabs,
            epsrel=epsrel,
        )
        return integral

    def __call__(self, x: Float[Array, "..."]) -> Float[Array, "..."]:
        return self.prob(x)


class EVMGaussian(EVMDistribution):
    """Gaussian distribution with mean and standard deviation as parameters."""
    mu: evm.Parameter = eqx.field(kw_only=True)
    sigma: evm.Parameter = eqx.field(kw_only=True)

    def unnormalized_prob(self, value):
        return jnp.exp(
            -0.5 * ((value - self.mu.value) / self.sigma.value) ** 2
        )
    
    def sample(self, sample_shape, seed=None, **kwargs):
        # Use JAX to sample from a normal distribution
        return jax.random.normal(
            seed, shape=sample_shape, dtype=self.mu.value.dtype
        ) * self.sigma.value + self.mu.value


class EVMExponential(EVMDistribution):
    lambd: evm.Parameter = eqx.field(kw_only=True)

    def unnormalized_prob(self, value):
        return jnp.exp(-self.lambd.value * value)

    def sample(self, sample_shape, seed=None, **kwargs):
        lower = self.var.lower
        upper = self.var.upper
        """Sample from a truncated exponential between xmin and xmax."""
        u = jax.random.uniform(seed, shape=sample_shape)
        lambda_val = self.lambd.value

        # Compute inverse CDF of truncated exponential
        z = jnp.exp(-lambda_val * lower) - u * (jnp.exp(-lambda_val * lower) - jnp.exp(-lambda_val * upper))
        samples = -jnp.log(z) / lambda_val

        return samples

        
class EVMSumPDF(EVMDistribution):
    pdfs: list = eqx.field(kw_only=True)

    def __post_init__(self):
        super().__post_init__()
        total_extended = jnp.asarray(0.0, dtype=jnp.asarray(self.extended.value).dtype)
        modifier_params = list(self.modifier_parameters)
        seen = list(modifier_params)
        for pdf in self.pdfs:
            total_extended = total_extended + pdf.extended.value
            for param in pdf.modifier_parameters:
                if all(existing is not param for existing in seen):
                    seen.append(param)
                    modifier_params.append(param)
        object.__setattr__(
            self,
            "extended",
            eqx.tree_at(
                lambda p: p.value,
                self.extended,
                jnp.asarray(
                    total_extended,
                    dtype=jnp.asarray(self.extended.value).dtype,
                ),
            ),
        )
        self.modifier_parameters = tuple(modifier_params)

    # to check if correct
    def unnormalized_prob(self, x: Float[Array, "..."]) -> Float[Array, "..."]:
        #factors = [pdf.extended / self.extended for pdf in self.pdfs]
        #return jnp.sum(jnp.array([pdf.prob(x) * factor for pdf, factor in zip(self.pdfs, factors)]))
        
        #!!! Note: using this gives completely different results
        return jnp.sum(
            jnp.array(
                [pdf.extended.value / self.extended.value * pdf(x) for pdf in self.pdfs]
            ),
            axis=0,
        )
        #return sum(jnp.array([pdf.extended / self.extended * pdf(x) for pdf in self.pdfs]))

    def prob(self, x: Float[Array, "..."]) -> Float[Array, "..."]:
        #norm = self.integrate()
        #return self.unnormalized_prob(x) / norm
        return self.unnormalized_prob(x)
    
    def sample(self, key):
        samples = []
        for pdf in self.pdfs:
            n2 = jax.random.poisson(lam=pdf.extended.value, key=key, shape=())
            n2 = jnp.asarray(n2, dtype=jnp.int64)
            sample = pdf.sample(sample_shape=(n2,), seed=key)
            samples.append(sample)
        return jnp.concatenate(samples, axis=-1)

    def _sample(self, seed, sample_shape=(), **kwargs):
        # one key per toy
        keys = jax.random.split(seed, sample_shape[0])
        samples = []
        def make_toy(key):
        #for key in keys:
            component_idx = jax.random.choice(
                key,
                len(self.pdfs),
                (sample_shape[1],),
                p=self.weights.reshape(-1),
            )
            # sample from each component
            component_samples = []
            for idx in range(len(self.pdfs)):
                component_samples.append(
                    self.pdfs[idx].sample(sample_shape=sample_shape, seed=key, **kwargs).reshape(-1)
                )
            # select the samples from the chosen component
            component_samples = jnp.array(component_samples)
            # based on component_idx, make a new array that selects the events from the chosen component
            event_indices = jnp.arange(sample_shape[1])
            # debug: print total, how many events per component
            #print("How many 0s and 1s") 
            #print(jnp.sum(component_idx == 0))
            #print(jnp.sum(component_idx == 1))
            #print(len(component_idx))
            sample = component_samples[component_idx, event_indices]
            samples.append(sample)
            return sample
        samples = jax.vmap(make_toy)(keys)
        # reshape such that samples is of shape (number_of_toys, nevents)
        #samples = jnp.array(samples)

        # debug
        #for k in keys:
        #    samples.append(make_toy(k))
        #samples = jnp.array(samples)

        return samples
   

class ExtendedNLL(eqx.Module):
    #Generalized Extended Likelihood for a mixture of multiple PDFs.

    model: EVMDistribution  # The model to compute the likelihood for
    #nus: list  # List of event yields (expected event counts per PDF)

    #def __init__(self, pdfs, nus, constraints=None):
    #    assert len(pdfs) == len(nus), (
    #        "Number of PDFs must match number of event yields."
    #    )
    #    self.pdfs = pdfs
    #    self.nus = nus
    #    self.constraints = constraints if constraints is not None else []

    def __call__(self, x):
        N = x.shape[0]  # Number of observed events
        #nu_total = sum(nu.value for nu in self.nus)  # Total expected events
        nu_total = self.model.extended.value  # Total expected events from the model

        # Compute mixture PDF
        #pdf = sum(nu.value / nu_total * pdf(x) for pdf, nu in zip(self.pdfs, self.nus))
        pdf = self.model.prob(x)  # Probability density function for the model

        # Extended likelihood calculation
        poisson_term = -nu_total + N * jnp.log(nu_total)  # Log(Poisson term)
        log_likelihood = poisson_term + jnp.sum(
            jnp.log(pdf + 1e-8)
        )  # Log-likelihood sum

        # Add auxiliary terms from parameter priors (if present)
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

        return jnp.squeeze(-log_likelihood)  # Negative log-likelihood for minimization
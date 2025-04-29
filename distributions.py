import jax
import jax.numpy as jnp
from jax import jit
from jax import lax
from functools import partial
import oryx
import evermore as evm
import equinox as eqx


class ExtendedNLL(eqx.Module):
    """Generalized Extended Likelihood for a mixture of multiple PDFs."""

    pdfs: list
    nus: list  # List of event yields (expected event counts per PDF)
    constraints: list

    def __init__(self, pdfs, nus, constraints=None):
        assert len(pdfs) == len(nus), (
            "Number of PDFs must match number of event yields."
        )
        self.pdfs = pdfs
        self.nus = nus
        self.constraints = constraints if constraints is not None else []

    def __call__(self, x):
        N = x.shape[0]  # Number of observed events
        nu_total = sum(nu.value for nu in self.nus)  # Total expected events

        # Compute mixture PDF
        pdf = sum(nu.value / nu_total * pdf(x) for pdf, nu in zip(self.pdfs, self.nus))

        # Extended likelihood calculation
        poisson_term = -nu_total + N * jnp.log(nu_total)  # Log(Poisson term)
        log_likelihood = poisson_term + jnp.sum(
            jnp.log(pdf + 1e-8)
        )  # Log-likelihood sum

        # add constraints
        constraint_term = sum(c() for c in self.constraints)
        log_likelihood += constraint_term

        return jnp.squeeze(-log_likelihood)  # Negative log-likelihood for minimization


class GaussianConstraint:
    def __init__(self, param, mu, sigma):
        self.param = param
        self.mu = mu
        self.sigma = sigma

    def __call__(self):
        return -0.5 * ((self.param.value - self.mu) / self.sigma) ** 2


class EVMExponential(oryx.distributions.Distribution):
    def __init__(self, lambd):
        self.lambd = lambd

    def prob(self, x):
        xmax = jnp.max(x)
        xmin = jnp.min(x)
        factor = self.lambd.value / (
            jnp.exp(-self.lambd.value * xmin) - jnp.exp(-self.lambd.value * xmax)
        )
        return factor * jnp.exp(-self.lambd.value * x)

    def log_prob(self, x):
        return jnp.log(self.prob(x))

    def __call__(self, x):
        return self.prob(x)

    def sample(self, sample_shape, seed=None, xmin=0., xmax=10., **kwargs):
        """Sample from a truncated exponential between xmin and xmax."""
        u = jax.random.uniform(seed, shape=sample_shape)
        lambda_val = self.lambd.value

        # Compute inverse CDF of truncated exponential
        z = jnp.exp(-lambda_val * xmin) - u * (jnp.exp(-lambda_val * xmin) - jnp.exp(-lambda_val * xmax))
        samples = -jnp.log(z) / lambda_val

        return samples


class EVMGaussian(oryx.distributions.Distribution):
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def prob(self, x):
        return (1 / (self.sigma.value * jnp.sqrt(2 * jnp.pi))) * jnp.exp(
            -0.5 * ((x - self.mu.value) / self.sigma.value) ** 2
        )

    def log_prob(self, x):
        return jnp.log(self.prob(x))

    def __call__(self, x):
        return self.prob(x)

    def sample(self, sample_shape, seed=None, **kwargs):
        # Use JAX to sample from a normal distribution
        return jax.random.normal(
            seed, shape=sample_shape, dtype=self.mu.value.dtype
        ) * self.sigma.value + self.mu.value

        
class EVMMixture(oryx.distributions.Distribution):
    def __init__(self, components, weights):
        self.components = components
        self.weights = jnp.array([p.value for p in weights]) / jnp.sum(jnp.array([p.value for p in weights]))

    def prob(self, x):
        return sum(w * comp.prob(x) for comp, w in zip(self.components, self.weights))

    def log_prob(self, x):
        return jnp.log(self.prob(x))

    def __call__(self, x):
        return self.prob(x)

    def sample(self, seed, sample_shape=(), **kwargs):
        # one key per toy
        keys = jax.random.split(seed, sample_shape[0])
        samples = []
        def make_toy(key):
        #for key in keys:
            component_idx = jax.random.choice(
                key,
                len(self.components),
                (sample_shape[1],),
                p=self.weights.reshape(-1),
            )
            # sample from each component
            component_samples = []
            for idx in range(len(self.components)):
                component_samples.append(
                    self.components[idx].sample(sample_shape=sample_shape, seed=key, **kwargs).reshape(-1)
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
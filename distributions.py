import jax
import jax.numpy as jnp
from jax import jit
from jax import lax
from functools import partial
import oryx
import evermore as evm


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
            print(component_idx)
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
            sample = component_samples[component_idx, event_indices]
            samples.append(sample)
            return sample
        samples = jax.vmap(make_toy)(keys)
        # reshape such that samples is of shape (number_of_toys, nevents)
        #samples = jnp.array(samples)
        return samples
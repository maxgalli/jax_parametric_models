import jax.numpy as jnp
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

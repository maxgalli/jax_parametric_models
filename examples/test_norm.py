import jax
import jax.numpy as jnp
from jax import jit
from jax import lax
from functools import partial
import oryx
import evermore as evm
import equinox as eqx
from quadax import romberg, quadgk


class Gauss:
    def __init__(self, mean, stddev):
        self.mean = mean
        self.stddev = stddev
        self.lower_bound = 100.
        self.upper_bound = 180.
        self.norm = self.integrate()

    def unnormalized_prob(self, value):
        return jnp.exp(-0.5 * ((value - self.mean) / self.stddev) ** 2)

    def integrate(self, lower=None, upper=None):
        if lower is None:
            lower = self.lower_bound
        if upper is None:
            upper = self.upper_bound

        epsabs = epsrel = 1e-5
        integral, _ = romberg(
            self.unnormalized_prob,
            [lower, upper],
            epsabs=epsabs,
            epsrel=epsrel,
        )
        print("diocane")
        print(integral)
        return integral

    def prob(self, value):
        #norm = self.integrate()
        return self.unnormalized_prob(value) / self.norm

model = Gauss(mean=125., stddev=2.)
epsabs = epsrel = 1e-5
integral, _ = romberg(
    model.prob,
    [model.lower_bound, model.upper_bound],
    epsabs=epsabs,
    epsrel=epsrel,
)
print("Integral:", integral)
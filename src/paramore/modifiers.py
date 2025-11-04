from __future__ import annotations

import abc

import jax.numpy as jnp
import evermore as evm
from .distributions import EVMDistribution

__all__ = ["Modifier", "SymmLogNormalModifier"]


class Modifier(abc.ABC):
    """Base class for modifiers that act on evermore distributions."""

    def __init__(self, parameter: evm.Parameter) -> None:
        self.parameter = parameter

    @abc.abstractmethod
    def apply(self, distribution: EVMDistribution) -> EVMDistribution:
        """Return a modified copy of the distribution."""
        raise NotImplementedError


class SymmLogNormalModifier(Modifier):
    """Symmetric log-normal modifier scaling the expected yield."""

    def __init__(self, parameter: evm.Parameter, kappa: float) -> None:
        super().__init__(parameter)
        self.kappa = float(kappa)

    def apply(self, distribution: EVMDistribution) -> EVMDistribution:
        modifier_value = jnp.exp(jnp.log(self.kappa) * self.parameter.value)

        modifier_params = distribution.modifier_parameters
        if all(existing is not self.parameter for existing in modifier_params):
            modifier_params = modifier_params + (self.parameter,)

        distribution.modifier_parameters = modifier_params
        distribution.extended = distribution.extended.replace(
            value=distribution.extended.value * modifier_value
        )
        return distribution

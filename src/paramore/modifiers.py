from __future__ import annotations

import abc
from typing import Callable

import jax.numpy as jnp
import evermore as evm
import equinox as eqx

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
        modifier = self.kappa ** self.parameter.value
        #modifier = jnp.exp(jnp.log(self.kappa) * self.parameter.value)
        return eqx.tree_at(
            lambda dist: dist.extended,
            distribution,
            distribution.extended * modifier,
        )

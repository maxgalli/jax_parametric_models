from __future__ import annotations

import abc

import jax.numpy as jnp
import evermore as evm
from .distributions import Distribution

__all__ = ["Modifier", "SymmLogNormalModifier", "AsymmetricLogNormalModifier"]


class Modifier(abc.ABC):
    """Base class for modifiers that act on evermore distributions."""

    def __init__(self, parameter: evm.Parameter) -> None:
        self.parameter = parameter

    @abc.abstractmethod
    def apply(self, distribution: Distribution) -> Distribution:
        """Return a modified copy of the distribution."""
        raise NotImplementedError


class SymmLogNormalModifier(Modifier):
    """Symmetric log-normal modifier scaling the expected yield."""

    def __init__(self, parameter: evm.Parameter, kappa: float) -> None:
        super().__init__(parameter)
        self.kappa = float(kappa)

    def apply(self, distribution: Distribution) -> Distribution:
        modifier_value = self.kappa**self.parameter.value

        modifier_params = distribution.modifier_parameters
        if all(existing is not self.parameter for existing in modifier_params):
            modifier_params = modifier_params + (self.parameter,)

        distribution.modifier_parameters = modifier_params
        distribution.extended = distribution.extended.replace(
            value=distribution.extended.value * modifier_value
        )
        return distribution


class AsymmetricLogNormalModifier(Modifier):
    """Asymmetric log-normal modifier scaling the expected yield."""

    def __init__(self, parameter: evm.Parameter, kappa_up: float, kappa_down: float) -> None:
        super().__init__(parameter)
        self.kappa_up = float(kappa_up)
        self.kappa_down = float(kappa_down)

    def apply(self, distribution: Distribution) -> Distribution:
        value = self.parameter.value
        kappa_capital = 0.125 * (
            4.0 * jnp.log(self.kappa_up / self.kappa_down)
            + jnp.log(self.kappa_up * self.kappa_down)
            * (48.0 * value**5 - 40.0 * value**3 + 15.0 * value)
        )

        modifier_value = jnp.where(
            value < -0.5,
            self.kappa_down ** (-value),
            jnp.where(
                value > 0.5,
                self.kappa_up ** value,
                jnp.exp(value * kappa_capital),
            ),
        )

        modifier_params = distribution.modifier_parameters
        if all(existing is not self.parameter for existing in modifier_params):
            modifier_params = modifier_params + (self.parameter,)

        distribution.modifier_parameters = modifier_params
        distribution.extended = distribution.extended.replace(
            value=distribution.extended.value * modifier_value
        )
        return distribution

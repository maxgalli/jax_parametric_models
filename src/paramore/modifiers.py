from __future__ import annotations

import abc

import jax.numpy as jnp
import evermore as evm
from .distributions import Distribution, ParameterizedFunction

__all__ = ["Modifier", "SymmLogNormalModifier", "AsymmetricLogNormalModifier"]


class ScaledValue(ParameterizedFunction):
    """Wrapper to scale a parameter or function by a modifier."""

    def __init__(self, original, modifier):
        self.original = original
        self.modifier = modifier

    @property
    def value(self):
        # Compute modifier value on-the-fly
        modifier_value = self.modifier._compute_modifier_value()
        return self.original.value * modifier_value


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

    def _compute_modifier_value(self):
        return self.kappa**self.parameter.value

    def apply(self, distribution: Distribution) -> Distribution:
        distribution.extended = ScaledValue(distribution.extended, self)
        return distribution


class AsymmetricLogNormalModifier(Modifier):
    """Asymmetric log-normal modifier scaling the expected yield."""

    def __init__(self, parameter: evm.Parameter, kappa_up: float, kappa_down: float) -> None:
        super().__init__(parameter)
        self.kappa_up = float(kappa_up)
        self.kappa_down = float(kappa_down)

    def _compute_modifier_value(self):
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
        return modifier_value

    def apply(self, distribution: Distribution) -> Distribution:
        distribution.extended = ScaledValue(distribution.extended, self)
        return distribution

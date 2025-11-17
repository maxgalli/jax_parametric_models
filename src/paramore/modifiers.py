"""Modifiers for scaling expected event counts in statistical models."""

from __future__ import annotations

import jax.numpy as jnp
import evermore as evm
from .distributions import ParameterizedFunction

__all__ = ["ScaledValue", "SymmLogNormalModifier", "AsymmetricLogNormalModifier"]


class ScaledValue(ParameterizedFunction):
    """Wrapper that scales a ParameterizedFunction by a modifier.

    Takes a base ParameterizedFunction and a modifier, returns: base * modifier_value
    This enables composable modifiers on expected values.
    """

    def __init__(self, base: ParameterizedFunction, modifier):
        """Initialize ScaledValue.

        Args:
            base: The original ParameterizedFunction to be scaled
            modifier: A modifier object with _compute_modifier_value() method
        """
        self.base = base
        self.modifier = modifier

    @property
    def value(self):
        """Return the scaled value."""
        # Compute modifier value on-the-fly
        modifier_value = self.modifier._compute_modifier_value()
        return self.base.value * modifier_value


class SymmLogNormalModifier:
    """Symmetric log-normal modifier: value * kappa^alpha.

    Takes a nuisance parameter and kappa, scales expected values by kappa^param.
    """

    def __init__(self, parameter: evm.Parameter, kappa: float):
        """Initialize SymmLogNormalModifier.

        Args:
            parameter: Nuisance parameter (typically constrained by a Normal prior)
            kappa: Multiplicative factor (kappa > 1 for upward variation, kappa < 1 for downward)
        """
        self.parameter = parameter
        self.kappa = float(kappa)

    def _compute_modifier_value(self):
        """Compute kappa^alpha where alpha is the parameter value."""
        return jnp.power(self.kappa, self.parameter.value)

    def apply(self, base: ParameterizedFunction) -> ParameterizedFunction:
        """Apply modifier to a ParameterizedFunction, returning scaled version.

        Args:
            base: ParameterizedFunction to be modified

        Returns:
            ScaledValue wrapping the base function
        """
        return ScaledValue(base, self)


class AsymmetricLogNormalModifier:
    """Asymmetric log-normal modifier: value * f(alpha, kappa_up, kappa_down).

    Uses different kappa values for up/down variations with smooth interpolation.
    """

    def __init__(self, parameter: evm.Parameter, kappa_up: float, kappa_down: float):
        """Initialize AsymmetricLogNormalModifier.

        Args:
            parameter: Nuisance parameter (typically constrained by a Normal prior)
            kappa_up: Multiplicative factor for positive parameter values
            kappa_down: Multiplicative factor for negative parameter values
        """
        self.parameter = parameter
        self.kappa_up = float(kappa_up)
        self.kappa_down = float(kappa_down)

    def _compute_modifier_value(self):
        """Compute modifier value with smooth interpolation.

        Uses a polynomial interpolation for |alpha| < 0.5 and
        simple power laws for |alpha| >= 0.5.
        """
        value = self.parameter.value

        # Smooth interpolation formula
        kappa_capital = 0.125 * (
            4.0 * jnp.log(self.kappa_up / self.kappa_down)
            + jnp.log(self.kappa_up * self.kappa_down)
            * (48.0 * value**5 - 40.0 * value**3 + 15.0 * value)
        )

        modifier_value = jnp.where(
            value < -0.5,
            jnp.power(self.kappa_down, -value),
            jnp.where(
                value > 0.5,
                jnp.power(self.kappa_up, value),
                jnp.exp(value * kappa_capital),
            ),
        )
        return modifier_value

    def apply(self, base: ParameterizedFunction) -> ParameterizedFunction:
        """Apply modifier to a ParameterizedFunction, returning scaled version.

        Args:
            base: ParameterizedFunction to be modified

        Returns:
            ScaledValue wrapping the base function
        """
        return ScaledValue(base, self)

"""Modifiers for scaling expected event counts in statistical models."""

from __future__ import annotations

import jax.numpy as jnp
import evermore as evm

__all__ = ["SymmLogNormalModifier", "AsymmetricLogNormalModifier", "ComposedModifier"]


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

    def apply(self, base: evm.Parameter) -> evm.Parameter:
        """Apply modifier to a Parameter, returning scaled Parameter.

        Args:
            base: Parameter to be modified

        Returns:
            New Parameter with scaled value
        """
        modifier_value = jnp.power(self.kappa, self.parameter.value)
        scaled_value = base.value * modifier_value

        return evm.Parameter(
            value=scaled_value,
            name=f"{base.name}_scaled_{self.parameter.name}",
            lower=None,
            upper=None,
            frozen=base.frozen,
        )


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

    def apply(self, base: evm.Parameter) -> evm.Parameter:
        """Apply modifier to a Parameter, returning scaled Parameter.

        Args:
            base: Parameter to be modified

        Returns:
            New Parameter with scaled value
        """
        modifier_value = self._compute_modifier_value()
        scaled_value = base.value * modifier_value

        return evm.Parameter(
            value=scaled_value,
            name=f"{base.name}_scaled_{self.parameter.name}",
            lower=None,
            upper=None,
            frozen=base.frozen,
        )


class ComposedModifier:
    """Compose multiple modifiers into a single modifier.

    Applies modifiers sequentially: the output of one modifier becomes
    the input to the next.
    """

    def __init__(self, *modifiers):
        """Initialize ComposedModifier.

        Args:
            *modifiers: Variable number of modifier objects, each with an apply() method
        """
        self.modifiers = modifiers

    def apply(self, base: evm.Parameter) -> evm.Parameter:
        """Apply all modifiers sequentially to a Parameter.

        Args:
            base: Parameter to be modified

        Returns:
            New Parameter with all modifiers applied
        """
        result = base
        for modifier in self.modifiers:
            result = modifier.apply(result)
        return result

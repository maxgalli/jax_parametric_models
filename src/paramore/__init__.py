"""Paramore: JAX-based parametric statistical modeling."""

import evermore as evm

# Workaround: evermore stores lower/upper as JAX arrays in pytree metadata,
# violating JAX's requirement that metadata be hashable with simple equality.
# See: https://docs.jax.dev/en/latest/custom_pytrees.html
_orig_base_param_init = evm.parameters.parameter.BaseParameter.__init__


def _patched_base_param_init(self, *args, **kwargs):
    _orig_base_param_init(self, *args, **kwargs)
    for key in ("lower", "upper"):
        val = self._var_metadata.get(key)
        if val is not None:
            self._var_metadata[key] = float(val)


evm.parameters.parameter.BaseParameter.__init__ = _patched_base_param_init

from .distributions import (
    BasePDF,
    Exponential,
    Gaussian,
    SumPDF,
)
from .likelihood import create_extended_nll, create_nll
from .modifiers import (
    AsymmetricLogNormalModifier,
    ComposedModifier,
    SymmLogNormalModifier,
)

__all__ = [
    # Distributions
    "BasePDF",
    "Gaussian",
    "Exponential",
    "SumPDF",
    # Modifiers
    "SymmLogNormalModifier",
    "AsymmetricLogNormalModifier",
    "ComposedModifier",
    # Likelihood
    "create_extended_nll",
    "create_nll",
]

"""Extended likelihood fitting example using paramore.

This example demonstrates:
1. Building PDFs with parameterized functions
2. Applying modifiers to expected event counts
3. Computing extended negative log-likelihood
4. Fitting to data and computing uncertainties
"""

from pathlib import Path

import evermore as evm
import jax
import jax.numpy as jnp
import optimistix
import pandas as pd
from evermore.parameters import filter as evm_filter
from evermore.parameters.transform import MinuitTransform, unwrap, wrap
from flax import nnx
from jax.experimental import checkify
from jax.flatten_util import ravel_pytree

# Import from paramore
import paramore as pm

wrap_checked = checkify.checkify(wrap)


# ============================================================================
# Parameters PyTree
# ============================================================================


class Params(nnx.Pytree):
    """Container for all parameters."""

    def __init__(
        self,
        mass,
        higgs_mass,
        d_higgs_mass,
        higgs_width,
        lamb,
        bkg_norm,
        mu,
        phoid_syst,
        jec_syst,
        nuisance_scale,
        nuisance_smear,
    ):
        self.mass = mass
        self.higgs_mass = higgs_mass
        self.d_higgs_mass = d_higgs_mass
        self.higgs_width = higgs_width
        self.lamb = lamb
        self.bkg_norm = bkg_norm
        self.mu = mu
        self.phoid_syst = phoid_syst
        self.jec_syst = jec_syst
        self.nuisance_scale = nuisance_scale
        self.nuisance_smear = nuisance_smear


# ============================================================================
# Main script
# ============================================================================

if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)

    minuit_transform = MinuitTransform()

    # Load data
    xs_ggH = 48.58  # pb
    br_hgg = 0.0027
    lumi = 138000.0
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "samples"
    fl = data_dir / "mc_part1.parquet"
    df = pd.read_parquet(fl)
    sumw = df["weight"].sum()
    eff = sumw / (xs_ggH * br_hgg)

    fl_data = data_dir / "data_part1.parquet"
    df_data = pd.read_parquet(fl_data)
    data = jnp.array(df_data["CMS_hgg_mass"].values)

    # ========================================================================
    # Define parameters
    # ========================================================================

    mass = evm.Parameter(
        value=125.0,
        name="CMS_hgg_mass",
        lower=100.0,
        upper=180.0,
        frozen=False,
    )

    params = Params(
        mass=mass,
        higgs_mass=evm.Parameter(
            value=125.0, name="higgs_mass", lower=120.0, upper=130.0, frozen=True
        ),
        d_higgs_mass=evm.Parameter(
            value=0.000848571,
            name="d_higgs_mass",
            lower=-5.0,
            upper=5.0,
            transform=minuit_transform,
            frozen=True,
        ),
        higgs_width=evm.Parameter(
            value=1.99705,
            name="higgs_width",
            lower=1.0,
            upper=5.0,
            transform=minuit_transform,
            frozen=True,
        ),
        lamb=evm.Parameter(
            value=0.1, name="lamb", lower=0.0, upper=1.0, transform=minuit_transform
        ),
        bkg_norm=evm.Parameter(
            value=float(df_data.shape[0]),
            name="bkg_norm",
            lower=0.0,
            upper=1e6,
            transform=minuit_transform,
        ),
        mu=evm.Parameter(
            value=1.0, name="mu", lower=0.0, upper=10.0, transform=minuit_transform
        ),
        phoid_syst=evm.NormalParameter(
            value=0.0, name="phoid_syst", transform=minuit_transform
        ),
        jec_syst=evm.NormalParameter(
            value=0.0, name="jec_syst", transform=minuit_transform
        ),
        nuisance_scale=evm.NormalParameter(
            value=0.0,
            name="nuisance_scale",
            lower=-5.0,
            upper=5.0,
            transform=minuit_transform,
        ),
        nuisance_smear=evm.NormalParameter(
            value=0.0,
            name="nuisance_smear",
            lower=-5.0,
            upper=5.0,
            transform=minuit_transform,
        ),
    )

    # ========================================================================
    # Fit
    # ========================================================================

    print("Fitting model to data...")

    # Unwrap and split for optimization
    params_unwrapped = unwrap(params)
    graphdef, diffable, static = nnx.split(
        params_unwrapped, evm_filter.is_dynamic_parameter, ...
    )

    @nnx.jit
    def loss_fn(dynamic_state, args):
        """Extended NLL computed using paramore."""
        graphdef, static_state, data, mass, xs_ggH, br_hgg, eff, lumi = args

        # Reconstruct params
        params_unwrapped_local = nnx.merge(
            graphdef, dynamic_state, static_state, copy=True
        )
        errors, params_wrapped = wrap_checked(params_unwrapped_local)

        # ====================================================================
        # Compute signal PDF parameters directly
        # ====================================================================
        signal_mu = (
            params_wrapped.higgs_mass.get_value()
            + params_wrapped.d_higgs_mass.get_value()
        ) * (1.0 + 0.003 * params_wrapped.nuisance_scale.get_value())
        signal_sigma = params_wrapped.higgs_width.get_value() * (
            1.0 + 0.045 * params_wrapped.nuisance_smear.get_value()
        )

        # Create signal PDF
        signal_pdf = pm.Gaussian(
            mu=signal_mu,
            sigma=signal_sigma,
            lower=mass.lower,
            upper=mass.upper,
        )

        # ====================================================================
        # Compute signal expected count using evm.Parameter + Modifiers
        # ====================================================================
        signal_rate_base = evm.Parameter(
            value=params_wrapped.mu.get_value() * xs_ggH * br_hgg * eff * lumi,
            name="signal_rate_base",
        )

        # Apply modifiers
        phoid_modifier = pm.SymmLogNormalModifier(
            parameter=params_wrapped.phoid_syst, kappa=1.05
        )

        jec_modifier = pm.AsymmetricLogNormalModifier(
            parameter=params_wrapped.jec_syst,
            kappa_up=1.056,
            kappa_down=0.951,
        )
        composed_modifier = pm.ComposedModifier(phoid_modifier, jec_modifier)
        signal_rate_with_all_modifiers = composed_modifier.apply(signal_rate_base)

        signal_rate = signal_rate_with_all_modifiers.get_value()

        # ====================================================================
        # Create background PDF
        # ====================================================================
        background_pdf = pm.Exponential(
            lambd=params_wrapped.lamb.get_value(),
            lower=mass.lower,
            upper=mass.upper,
        )

        bkg_rate = params_wrapped.bkg_norm.get_value()

        # ====================================================================
        # Compute extended NLL using paramore
        # ====================================================================
        sum_pdf = pm.SumPDF(
            pdfs=[signal_pdf, background_pdf],
            extended_vals=[signal_rate, bkg_rate],
            lower=mass.lower,
            upper=mass.upper,
        )

        nll = pm.create_extended_nll(params_wrapped, sum_pdf, data)

        return jnp.squeeze(nll)

    # Optimize
    solver = optimistix.BFGS(rtol=1e-5, atol=1e-7)
    fitresult = optimistix.minimise(
        loss_fn,
        solver,
        diffable,
        has_aux=False,
        args=(graphdef, static, data, mass, xs_ggH, br_hgg, eff, lumi),
        options={},
        max_steps=1000,
        throw=False,
    )

    # Extract results
    fitted_unwrapped = nnx.merge(graphdef, fitresult.value, static, copy=True)
    fitted_params = wrap(fitted_unwrapped)

    # ========================================================================
    # Compute uncertainties using inverse Hessian from BFGS
    # ========================================================================

    hessian_inv_op = fitresult.state.f_info.hessian_inv
    if hessian_inv_op is None:
        raise ValueError("No inverse Hessian available from optimizer")

    # Flatten parameters and construct covariance matrix
    flat_opt, unravel = ravel_pytree(fitresult.value)
    cov_matrix = jnp.asarray(hessian_inv_op.as_matrix(), dtype=flat_opt.dtype)

    def param_uncertainty(selector):
        """Propagate uncertainties from diffable space to a physical parameter.

        Args:
            selector: Function that takes params and returns the parameter of interest

        Returns:
            Standard deviation of the parameter
        """

        def value_fn(flat_params):
            diffable_params = unravel(flat_params)
            params_unwrapped_local = nnx.merge(
                graphdef, diffable_params, static, copy=True
            )
            _, params_wrapped = wrap_checked(params_unwrapped_local)
            return selector(params_wrapped)

        # Compute gradient
        grad = jax.grad(value_fn)(flat_opt)

        # Error propagation
        variance = jnp.dot(grad, cov_matrix @ grad)
        return jnp.sqrt(variance)

    # Compute uncertainties
    mu_sigma = param_uncertainty(lambda p: p.mu.get_value())
    bkg_norm_sigma = param_uncertainty(lambda p: p.bkg_norm.get_value())
    lamb_sigma = param_uncertainty(lambda p: p.lamb.get_value())
    phoid_sigma = param_uncertainty(lambda p: p.phoid_syst.get_value())
    jec_sigma = param_uncertainty(lambda p: p.jec_syst.get_value())
    nuisance_scale_sigma = param_uncertainty(lambda p: p.nuisance_scale.get_value())
    nuisance_smear_sigma = param_uncertainty(lambda p: p.nuisance_smear.get_value())

    print("\n" + "=" * 60)
    print("Fit Results:")
    print("=" * 60)
    print(f"r = {float(fitted_params.mu.get_value()):.6f} ± {float(mu_sigma):.6f}")
    print(
        f"bkg_norm = {float(fitted_params.bkg_norm.get_value()):.2f} ± {float(bkg_norm_sigma):.2f}"
    )
    print(
        f"lamb = {float(fitted_params.lamb.get_value()):.6f} ± {float(lamb_sigma):.6f}"
    )
    print(
        f"phoid_syst = {float(fitted_params.phoid_syst.get_value()):.6f} ± {float(phoid_sigma):.6f}"
    )
    print(
        f"jec_syst = {float(fitted_params.jec_syst.get_value()):.6f} ± {float(jec_sigma):.6f}"
    )
    print(
        f"nuisance_scale = {float(fitted_params.nuisance_scale.get_value()):.6f} ± {float(nuisance_scale_sigma):.6f}"
    )
    print(
        f"nuisance_smear = {float(fitted_params.nuisance_smear.get_value()):.6f} ± {float(nuisance_smear_sigma):.6f}"
    )
    print("=" * 60)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
from scipy import interpolate
from pathlib import Path
import jax.numpy as jnp
import evermore as evm
from flax import nnx
import jax
import optax
from evermore.parameters.transform import MinuitTransform, unwrap, wrap
from evermore.parameters import filter as evm_filter
from jax.experimental import checkify
import optimistix
from jax.flatten_util import ravel_pytree


from paramore import (
    Exponential,
    Gaussian,
    ExtendedNLL,
    SumPDF,
    ParameterizedFunction,
    plot_as_data,
    save_image,
)
from paramore.modifiers import SymmLogNormalModifier, AsymmetricLogNormalModifier


wrap_checked = checkify.checkify(wrap)


class Params(nnx.Pytree):
    def __init__(
        self,
        higgs_mass: evm.Parameter,
        d_higgs_mass: evm.Parameter,
        higgs_width: evm.Parameter,
        lamb: evm.Parameter,
        bkg_norm: evm.Parameter,
        mu: evm.Parameter,
        phoid_syst: evm.NormalParameter,
        jec_syst: evm.NormalParameter,
        nuisance_scale: evm.Parameter,
        nuisance_smear: evm.Parameter,
    ) -> None:
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


# double precision
jax.config.update("jax_enable_x64", True)

# plot styling
hep.style.use("CMS")


if __name__ == "__main__":
    minuit_transform = MinuitTransform()

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

    data = jax.numpy.array(df_data["CMS_hgg_mass"].values)

    # variable for pdf, the mass
    true_mean = 125.0
    mass = evm.Parameter(
        value=true_mean,
        name="CMS_hgg_mass",
        lower=100.0,
        upper=180.0,
        frozen=False,
    )

    class MeanFunctionWithScale(ParameterizedFunction):
        """Compute mean with scale nuisance parameter."""

        def __init__(
            self,
            higgs_mass: evm.Parameter,
            d_higgs_mass: evm.Parameter,
            nuisance_scale: evm.Parameter,
        ) -> None:
            self.higgs_mass = higgs_mass
            self.d_higgs_mass = d_higgs_mass
            self.nuisance_scale = nuisance_scale

        @property
        def value(self):
            return (self.higgs_mass.value + self.d_higgs_mass.value) * (
                1.0 + 0.003 * self.nuisance_scale.value
            )

    class SigmaFunctionWithSmear(ParameterizedFunction):
        """Compute sigma with smear nuisance parameter."""

        def __init__(
            self,
            sigma: evm.Parameter,
            nuisance_smear: evm.Parameter,
        ) -> None:
            self.sigma = sigma
            self.nuisance_smear = nuisance_smear

        @property
        def value(self):
            return self.sigma.value * (1.0 + 0.045 * self.nuisance_smear.value)

    class SignalRate(ParameterizedFunction):
        """Compute signal rate from signal strength and constants."""

        def __init__(
            self,
            mu: evm.Parameter,
            xs_ggH: float,
            br_hgg: float,
            eff: float,
            lumi: float,
        ) -> None:
            self.mu = mu
            self.xs_ggH = xs_ggH
            self.br_hgg = br_hgg
            self.eff = eff
            self.lumi = lumi

        @property
        def value(self):
            return self.mu.value * self.xs_ggH * self.br_hgg * self.eff * self.lumi

    params = Params(
        evm.Parameter(
            value=125.0, name="higgs_mass", lower=120.0, upper=130.0, frozen=True
        ),
        evm.Parameter(
            value=0.000848571,
            name="d_higgs_mass",
            lower=-5.0,
            upper=5.0,
            transform=minuit_transform,
            frozen=True,
        ),
        evm.Parameter(
            value=1.99705,
            name="higgs_width",
            lower=1.0,
            upper=5.0,
            transform=minuit_transform,
            frozen=True,
        ),
        evm.Parameter(
            value=0.1,
            name="lamb",
            lower=0.0,
            upper=1.0,
            transform=minuit_transform,
        ),
        evm.Parameter(
            value=float(df_data.shape[0]),
            name="bkg_norm",
            lower=0.0,
            upper=1e6,
            transform=minuit_transform,
        ),
        evm.Parameter(
            value=1.0,
            name="mu",
            lower=0.0,
            upper=10.0,
            transform=minuit_transform,
        ),
        evm.NormalParameter(
            value=0.0, name="phoid_syst", transform=minuit_transform
        ),
        evm.NormalParameter(
            value=0.0, name="jec_syst", transform=minuit_transform
        ),
        evm.NormalParameter(
            value=0.0,
            name="nuisance_scale",
            lower=-5.0,
            upper=5.0,
            transform=minuit_transform,
        ),
        evm.NormalParameter(
            value=0.0,
            name="nuisance_smear",
            lower=-5.0,
            upper=5.0,
            transform=minuit_transform,
        )
    )

    def build_model(params):
        signal_pdf = Gaussian(
            var=mass,
            mu=MeanFunctionWithScale(
                params.higgs_mass, params.d_higgs_mass, params.nuisance_scale
            ),
            sigma=SigmaFunctionWithSmear(
                params.higgs_width, params.nuisance_smear
            ),
            extended=SignalRate(
                params.mu, xs_ggH, br_hgg, eff, lumi
            ),
        )
        pho_id_modifier = SymmLogNormalModifier(
            parameter=params.phoid_syst,
            kappa=1.05,
        )
        jec_modifier = AsymmetricLogNormalModifier(
            parameter=params.jec_syst,
            kappa_up=1.056,
            kappa_down=0.951,
        )
        signal_pdf = pho_id_modifier.apply(signal_pdf)
        signal_pdf = jec_modifier.apply(signal_pdf)

        bkg_pdf = Exponential(
            var=mass,
            lambd=params.lamb,
            extended=params.bkg_norm,
        )
        model = SumPDF(
            var=mass,
            pdfs=[signal_pdf, bkg_pdf],
        )
        return model

    params_unwrapped = unwrap(params)
    graphdef, diffable, static = nnx.split(
        params_unwrapped, evm_filter.is_dynamic_parameter, ...
    )

    def loss_fn(dynamic_state, args):
        graphdef, static_state, data = args
        params_unwrapped = nnx.merge(
            graphdef, dynamic_state, static_state, copy=True
        )
        _, params_wrapped = wrap_checked(params_unwrapped)
        model = build_model(params_wrapped)
        nll = ExtendedNLL(model=model)
        return nll(data)

    solver = optimistix.BFGS(
        rtol=1e-5, atol=1e-7, verbose=frozenset({"step_size", "loss"})
    )
    fitresult = optimistix.minimise(
        loss_fn,
        solver,
        diffable,
        has_aux=False,
        args=(graphdef, static, data),
        options={},
        max_steps=1000,
        throw=True,
    )
    fitted_unwrapped = nnx.merge(
        graphdef, fitresult.value, static, copy=True
    )
    fitted_params = wrap(fitted_unwrapped)

    # Extract inverse Hessian approximation from BFGS and turn it into a covariance.
    hessian_inv_op = fitresult.state.f_info.hessian_inv
    if hessian_inv_op is None:
        raise ValueError(
            "No inverse Hessian available"
        )

    flat_opt, unravel = ravel_pytree(fitresult.value)
    cov_matrix = jnp.asarray(hessian_inv_op.as_matrix(), dtype=flat_opt.dtype)

    def param_uncertainty(selector):
        """Propagate uncertainties from diffable space to a physical parameter."""

        def value_fn(flat_params):
            diffable_params = unravel(flat_params)
            params_unwrapped = nnx.merge(
                graphdef, diffable_params, static, copy=True
            )
            _, params_wrapped = wrap_checked(params_unwrapped)
            return selector(params_wrapped)

        grad = jax.grad(value_fn)(flat_opt)
        variance = jnp.dot(grad, cov_matrix @ grad)
        return jnp.sqrt(variance)

    mu_sigma = param_uncertainty(lambda p: p.mu.value)
    bkg_norm_sigma = param_uncertainty(lambda p: p.bkg_norm.value)
    lamb_sigma = param_uncertainty(lambda p: p.lamb.value)
    phoid_sigma = param_uncertainty(lambda p: p.phoid_syst.value)
    jec_sigma = param_uncertainty(lambda p: p.jec_syst.value)
    nuisance_scale_sigma = param_uncertainty(lambda p: p.nuisance_scale.value)
    nuisance_smear_sigma = param_uncertainty(lambda p: p.nuisance_smear.value)

    print(
        f"Final estimate: r = {float(fitted_params.mu.value):.6f} ± {float(mu_sigma):.6f}\n"
    )
    print(
        f"Final estimate: bkg_norm = {float(fitted_params.bkg_norm.value):.6f} ± {float(bkg_norm_sigma):.6f}\n"
    )
    print(
        f"Final estimate: lamb = {float(fitted_params.lamb.value):.6f} ± {float(lamb_sigma):.6f}\n"
    )
    print(
        f"Final estimate: phoid_syst = {float(fitted_params.phoid_syst.value):.6f} ± {float(phoid_sigma):.6f}\n"
    )
    print(
        f"Final estimate: jec_syst = {float(fitted_params.jec_syst.value):.6f} ± {float(jec_sigma):.6f}\n"
    )
    print(
        f"Final estimate: nuisance_scale = {float(fitted_params.nuisance_scale.value):.6f} ± {float(nuisance_scale_sigma):.6f}\n"
    )
    print(
        f"Final estimate: nuisance_smear = {float(fitted_params.nuisance_smear.value):.6f} ± {float(nuisance_smear_sigma):.6f}\n"
    )

    print("Scanning NLL vs mu...")

    denominator = loss_fn(fitresult.value, (graphdef, static, data))

    def fixed_mu_fit(mu, silent=True, params_template=params):
        params_copy = nnx.merge(*nnx.split(params_template), copy=True)
        params_copy.mu = params_copy.mu.replace(
            value=jnp.asarray(mu, dtype=params_copy.mu.value.dtype)
        )
        params_copy.mu.set_metadata(frozen=True)

        params_unwrapped_local = unwrap(params_copy)
        graphdef_local, diffable_local, static_local = nnx.split(
            params_unwrapped_local, evm_filter.is_dynamic_parameter, ...
        )

        fitresult_local = optimistix.minimise(
            loss_fn,
            solver,
            diffable_local,
            has_aux=False,
            args=(graphdef_local, static_local, data),
            options={},
            max_steps=1000,
            throw=True,
        )
        nll_val = loss_fn(
            fitresult_local.value, (graphdef_local, static_local, data)
        )

        if not silent:
            print(f"  mu = {mu:.3f} -> NLL = {nll_val:.3f}")

        return 2 * (nll_val - denominator)

    mu_values = jnp.linspace(0.01, 3.0, 20)
    nll_values = []
    for mu in mu_values:
        nll_values.append(fixed_mu_fit(mu))

    # plot
    print("Plotting NLL scan...")
    y = jnp.array(nll_values)
    x = jnp.array(mu_values)

    func = interpolate.interp1d(x, y, kind="cubic")
    n_interp = 1000
    x_interp = jnp.linspace(x[0], x[-1], n_interp)
    y_interp = func(x_interp)
    min_idx = int(jnp.argmin(y_interp))
    mu_hat = float(x_interp[min_idx])
    y_min = float(y_interp[min_idx])
    y_interp = y_interp - y_min

    def _find_crossings(target):
        diffs = y_interp - target
        sign_changes = jnp.where(jnp.diff(jnp.sign(diffs)))[0]
        if sign_changes.size == 0:
            return []
        roots = []
        for idx in sign_changes:
            x0, x1 = x_interp[idx], x_interp[idx + 1]
            y0, y1 = diffs[idx], diffs[idx + 1]
            roots.append(float(x0 - y0 * (x1 - x0) / (y1 - y0)))
        return roots

    crossings = _find_crossings(1.0)
    if len(crossings) >= 2:
        left_cross, right_cross = min(crossings), max(crossings)
        sigma_mu = 0.5 * (right_cross - left_cross)
    else:
        sigma_mu = None

    fig, ax = plt.subplots()
    ax.plot(x_interp, y_interp, label="NLL Scan", color="black")
    ax.set_xlabel("Signal strength $\\mu$")
    ax.set_ylabel("-2$\\Delta$NLL")
    ax.axhline(1.0, color="red", linestyle="--", label="$1\\sigma$ interval")
    ax.set_ylim(0., 10.)
    if sigma_mu is not None:
        ax.annotate(
            fr"$\hat{{\mu}} = {mu_hat:.3f}\pm{sigma_mu:.3f}$",
            xy=(mu_hat, 0.05),
            xycoords=("data", "axes fraction"),
            xytext=(10, 10),
            textcoords="offset points",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
        )
    ax.legend()
    output_dir = base_dir / "figures_api_example"
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "nll_scan.png")
    plt.savefig(output_dir / "nll_scan.pdf")
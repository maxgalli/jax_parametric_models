import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
from scipy import interpolate
from pathlib import Path
import jax.numpy as jnp
import evermore as evm
import equinox as eqx
from typing import NamedTuple, List, Set
import jax
import optax
from evermore.parameters.transform import MinuitTransform, unwrap, wrap
import optimistix
from jax.tree_util import tree_leaves
from jax.flatten_util import ravel_pytree


from paramore import (
    EVMExponential,
    EVMGaussian,
    ExtendedNLL,
    GaussianConstraint,
    EVMSumPDF,
    plot_as_data,
    save_image,
)
from paramore.modifiers import SymmLogNormalModifier


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

    def mean_function(higgs_mass, d_higgs_mass):
        return higgs_mass + d_higgs_mass

    def signal_rate(r, xs_ggH, br_hgg, eff, lumi):
        return r * xs_ggH * br_hgg * eff * lumi

    class Params(eqx.Module):
        # signal model
        higgs_mass: evm.Parameter
        d_higgs_mass: evm.Parameter
        higgs_width: evm.Parameter
        # bkg model
        lamb: evm.Parameter
        bkg_norm: evm.Parameter
        # signal rate
        mu: evm.Parameter
        # nuisances
        phoid_syst: evm.NormalParameter

    params = Params(
        higgs_mass=evm.Parameter(
            value=125.0, name="higgs_mass", lower=120.0, upper=130.0, frozen=True
        ),
        d_higgs_mass=evm.Parameter(
            #0.0, name="d_higgs_mass", lower=-1.0, upper=1.0, transform=minuit_transform, frozen=True
            0.000848571, name="d_higgs_mass", lower=-5.0, upper=5.0, transform=minuit_transform, frozen=True
        ),
        higgs_width=evm.Parameter(
            #2.0, name="higgs_width", lower=1.0, upper=5.0, transform=minuit_transform, frozen=True
            1.99705, name="higgs_width", lower=1.0, upper=5.0, transform=minuit_transform, frozen=True
        ),
        lamb=evm.Parameter(
            0.1, name="lamb", lower=0.0, upper=1.0, transform=minuit_transform
        ),
        bkg_norm=evm.Parameter(
            float(df_data.shape[0]),
            name="bkg_norm",
            lower=0.0,
            upper=1e6,
            transform=minuit_transform,
        ),
        mu=evm.Parameter(
            1.0, name="mu", lower=0.0, upper=10.0, transform=minuit_transform
        ),
        phoid_syst=evm.NormalParameter(
            value=0.0, name="phoid_syst", transform=minuit_transform
        ),
    )

    def build_model(params):
        signal_pdf = EVMGaussian(
            var=mass,
            mu=evm.Parameter(mean_function(params.higgs_mass.value, params.d_higgs_mass.value)),
            sigma=params.higgs_width,
            extended=evm.Parameter(signal_rate(params.mu.value, xs_ggH, br_hgg, eff, lumi))
        )
        pho_id_modifier = SymmLogNormalModifier(
            parameter=params.phoid_syst,
            kappa=1.05,
        )
        signal_pdf = pho_id_modifier.apply(signal_pdf)
        bkg_pdf = EVMExponential(
            var=mass,
            lambd=params.lamb,
            extended=params.bkg_norm,
        )
        model = EVMSumPDF(
            var=mass,
            pdfs=[signal_pdf, bkg_pdf],
        )
        return model

    model = build_model(params)
    nll = ExtendedNLL(model=model)
    nll_val = nll(data)
    #print(f"Initial NLL: {nll_val}")

    @eqx.filter_jit
    def loss(diffable, static, data):
        params = wrap(evm.tree.combine(diffable, static))
        model = build_model(params)
        nll = ExtendedNLL(model=model)
        p = wrap(params)
        #print(p.phoid_syst.value)
        return nll(data)

    diffable, static = evm.tree.partition(unwrap(params))
    #print(diffable)
    #print(static)

    def optx_loss_fn(diffable, args):
        return loss(diffable, *args)

    solver = optimistix.BFGS(
        rtol=1e-5, atol=1e-7, verbose=frozenset({"step_size", "loss"})
    )
    fitresult = optimistix.minimise(
        optx_loss_fn,
        solver,
        diffable,
        has_aux=False,
        args=(static, data),
        options={},
        max_steps=1000,
        throw=True,
    )
    fitted_params = wrap(evm.tree.combine(fitresult.value, static))

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
            params = wrap(evm.tree.combine(diffable_params, static))
            return selector(params)

        grad = jax.grad(value_fn)(flat_opt)
        variance = jnp.dot(grad, cov_matrix @ grad)
        return jnp.sqrt(variance)

    mu_sigma = param_uncertainty(lambda p: p.mu.value)
    bkg_norm_sigma = param_uncertainty(lambda p: p.bkg_norm.value)
    lamb_sigma = param_uncertainty(lambda p: p.lamb.value)
    phoid_sigma = param_uncertainty(lambda p: p.phoid_syst.value)

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
        f"Final estimatee: phoid_syst = {float(fitted_params.phoid_syst.value):.6f} ± {float(phoid_sigma):.6f}\n"
    )

"""
    print("Scanning NLL vs mu...")

    denominator = loss(fitresult.value, static, data)

    def fixed_mu_fit(mu, silent=True, params=params):
        params = eqx.tree_at(lambda p: p.mu.value, params, mu)
        params = eqx.tree_at(lambda p: p.mu.frozen, params, True)

        diffable, static = evm.tree.partition(unwrap(params))

        def optx_loss_fn(diffable, args):
            return loss(diffable, *args)

        fitresult = optimistix.minimise(
            optx_loss_fn,
            solver,
            diffable,
            has_aux=False,
            args=(static, data),
            options={},
            max_steps=1000,
            throw=True,
        )
        nll_val = loss(fitresult.value, static, data)

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
    y_interp = y_interp - jnp.min(y_interp)
    fig, ax = plt.subplots()
    ax.plot(x_interp, y_interp, label="NLL Scan", color="black")
    ax.set_xlabel(r"Signal strength $\mu$")
    ax.set_ylabel(r"-2$\Delta$NLL")
    ax.axhline(1.0, color="red", linestyle="--", label=r"$1\sigma$ interval")
    ax.set_ylim(-1.0, 10)
    ax.legend()
    output_dir = base_dir / "figures_api_example"
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "nll_scan.png")
    plt.savefig(output_dir / "nll_scan.pdf")
"""

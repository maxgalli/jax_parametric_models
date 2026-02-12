import os
from pathlib import Path
from typing import NamedTuple

import equinox as eqx
import evermore as evm
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import optimistix
import pandas as pd
from evermore.parameters.transform import MinuitTransform, unwrap, wrap
from flax import nnx
from jax.experimental import checkify
from plotting_helpers import plot_as_data, save_image
from scipy import interpolate

from paramore import Exponential, Gaussian, SumPDF, create_extended_nll, create_nll

# double precision
jax.config.update("jax_enable_x64", True)

wrap_checked = checkify.checkify(wrap)

# plot styling
hep.style.use("CMS")


if __name__ == "__main__":
    # Signal modelling
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "samples"
    fl = data_dir / "mc_part1.parquet"
    output_dir = base_dir / "figures_part1"
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_parquet(fl)
    var_name = "CMS_hgg_mass"

    fig, ax = plt.subplots()
    ax = plot_as_data(df[var_name], nbins=100, ax=ax)
    ax.set_xlabel(r"$m_{\gamma\gamma}$ [GeV]")
    save_image("part1_signal_mass", output_dir)

    data = jax.numpy.array(df["CMS_hgg_mass"].values)
    true_mean = 125.0

    minuit_transform = MinuitTransform()

    mass = evm.Parameter(
        value=true_mean,
        name="CMS_hgg_mass",
        lower=100.0,
        upper=180.0,
        frozen=False,
    )
    higgs_mass = evm.Parameter(
        value=125.0, name="higgs_mass", lower=120.0, upper=130.0, frozen=True
    )
    d_higgs_mass = evm.Parameter(
        value=0.0, name="dMH", lower=-1.0, upper=1.0, transform=minuit_transform
    )
    sigma = evm.Parameter(
        value=2.0, name="sigma", lower=1.0, upper=5.0, transform=minuit_transform
    )

    class Params(NamedTuple):
        higgs_mass: evm.Parameter
        d_higgs_mass: evm.Parameter
        sigma: evm.Parameter

    params = Params(higgs_mass, d_higgs_mass, sigma)

    def mean_function(higgs_mass, d_higgs_mass):
        return higgs_mass + d_higgs_mass

    # === Loss Function ===
    @eqx.filter_jit
    def loss_fn(diffable, static, data, graphdef):
        _, params = wrap_checked(nnx.merge(graphdef, diffable, static, copy=True))
        std = params.sigma
        composed_mu = evm.Parameter(
            mean_function(
                params.higgs_mass.get_value(), params.d_higgs_mass.get_value()
            )
        )

        model = Gaussian(
            mu=composed_mu,
            sigma=std,
            lower=mass.lower,
            upper=mass.upper,
        )

        # sumpdf = SumPDF(pdfs=[model], lower=mass.lower, upper=mass.upper)
        nll = create_nll(params, model, data)
        return jnp.squeeze(nll)

    graphdef, diffable, static = nnx.split(
        unwrap(params), evm.filter.is_dynamic_parameter, ...
    )

    def optx_loss_fn(diffable, args):
        return loss_fn(diffable, *args)

    solver = optimistix.BFGS(
        rtol=1e-5, atol=1e-7, verbose=frozenset({"step_size", "loss"})
    )
    fitresult = optimistix.minimise(
        optx_loss_fn,
        solver,
        diffable,
        has_aux=False,
        args=(static, data, graphdef),
        options={},
        max_steps=1000,
        throw=True,
    )
    fitted_params = wrap(nnx.merge(graphdef, fitresult.value, static, copy=True))

    # === Final Results ===
    print(f"Final estimate: Std = {fitted_params.sigma.get_value()}")
    print(f"Final estimate: dMH = {fitted_params.d_higgs_mass.get_value()}")

    # Signal normalisation
    print("Getting signal normalisation")
    xs_ggH = 48.58  # pb
    br_hgg = 0.0027

    # this part is slightly different from what jon does
    sumw = df["weight"].sum()
    eff = sumw / (xs_ggH * br_hgg)
    print(f"Efficiency of ggH events landing in Tag0 is: {eff:.5f}")

    lumi = 138000.0
    n = eff * xs_ggH * br_hgg * lumi
    print(
        f"For 138 fb^-1, the expected number of ggH events is: N = xs * BR * eff * lumi = {n:.5f}"
    )

    fl_data = data_dir / "data_part1.parquet"
    df_data = pd.read_parquet(fl_data)
    data_array = jax.numpy.array(df_data[var_name].values)
    var_name = "CMS_hgg_mass"
    # df_data_sides = df_data[(df_data[var_name] > 100) & (df_data[var_name] < 115) | (df_data[var_name] > 135) & (df_data[var_name] < 180)]
    # keep only data between 100 and 180
    df_data = df_data[(df_data[var_name] > 100) & (df_data[var_name] < 180)]
    # data_sides = jax.numpy.array(df_data_sides[var_name].values)
    data_sides = jax.numpy.array(df_data[var_name].values)
    lam = evm.Parameter(
        value=0.05, name="lambda", lower=0, upper=0.2, transform=minuit_transform
    )

    class ParamsBkg(NamedTuple):
        lambd: evm.Parameter

    params_bkg = ParamsBkg(lam)

    @eqx.filter_jit
    def loss_fn_bkg(diffable, static, data, graphdef):
        _, params = wrap_checked(nnx.merge(graphdef, diffable, static, copy=True))
        model = Exponential(
            lambd=params.lambd.get_value(),
            lower=mass.lower,
            upper=mass.upper,
        )
        nll = create_nll(params, model, data)
        return jnp.squeeze(nll)

    graphdef, diffable, static = nnx.split(
        unwrap(params_bkg), evm.filter.is_dynamic_parameter, ...
    )

    def optx_loss_fn(diffable, args):
        return loss_fn_bkg(diffable, *args)

    fitresult = optimistix.minimise(
        optx_loss_fn,
        solver,
        diffable,
        has_aux=False,
        args=(static, data_sides, graphdef),
        options={},
        max_steps=1000,
        throw=True,
    )
    fitted_params = wrap(nnx.merge(graphdef, fitresult.value, static, copy=True))

    # === Final Results ===
    print(f"Final estimate: Lambda = {fitted_params.lambd.get_value()}")

    fig, ax = plt.subplots()
    ax.set_xlabel(r"$m_{\gamma\gamma}$ [GeV]")
    x = np.linspace(100, 180, 1000)
    model_bkg = Exponential(
        lambd=fitted_params.lambd.get_value(), lower=mass.lower, upper=mass.upper
    )
    y = model_bkg.prob(x)
    ax.plot(x, y, label="fit")
    ax.legend()
    save_image("part1_data_sidebands", output_dir)

    ################
    # starting part 2
    ################
    xs_ggH_par = evm.Parameter(
        value=xs_ggH, name="xs_ggH", lower=0.0, upper=100.0, frozen=True
    )
    br_hgg_par = evm.Parameter(
        value=br_hgg, name="br_hgg", lower=0.0, upper=1.0, frozen=True
    )
    eff_par = evm.Parameter(value=eff, name="eff", lower=0.0, upper=1.0, frozen=True)
    lumi_par = evm.Parameter(
        value=lumi, name="lumi", lower=0.0, upper=1000000.0, frozen=True
    )
    r = evm.Parameter(
        value=1.0, name="r", lower=0.0, upper=20.0, transform=minuit_transform
    )

    def model_ggH_Tag0_norm_function(r, xs_ggH, br_hgg, eff, lumi):
        return r * xs_ggH * br_hgg * eff * lumi

    norm_bkg = evm.Parameter(
        value=float(df_data.shape[0]),
        name="model_bkg_Tag0_norm",
        lower=0.0,
        upper=float(3 * df_data.shape[0]),
        transform=minuit_transform,
    )

    def total_rate(r, xs_ggH, br_hgg, eff, lumi, model_bkg_norm):
        return r * xs_ggH * br_hgg * eff * lumi + model_bkg_norm

    class ParamsCard(NamedTuple):
        higgs_mass: evm.Parameter
        d_higgs_mass: evm.Parameter
        sigma: evm.Parameter
        lambd: evm.Parameter
        r: evm.Parameter
        xs_ggH: evm.Parameter
        br_hgg: evm.Parameter
        eff: evm.Parameter
        lumi: evm.Parameter
        model_bkg_norm: evm.Parameter

    # redefine sigma and d_higgs_mass with the best value from before such that they are now frozen
    sigma = evm.Parameter(
        value=params.sigma.get_value(), name="sigma", lower=1.0, upper=5.0, frozen=True
    )
    d_higgs_mass = evm.Parameter(
        value=params.d_higgs_mass.get_value(),
        name="dMH",
        lower=-1.0,
        upper=1.0,
        frozen=True,
    )
    params_card = ParamsCard(
        higgs_mass,
        d_higgs_mass,
        sigma,
        lam,
        r,
        xs_ggH_par,
        br_hgg_par,
        eff_par,
        lumi_par,
        norm_bkg,
    )

    # === Loss Function ===
    @eqx.filter_jit
    def loss_fn_card(diffable, static, data, graphdef):
        _, params = wrap_checked(nnx.merge(graphdef, diffable, static, copy=True))
        signal_rate = evm.Parameter(
            model_ggH_Tag0_norm_function(
                params.r.get_value(),
                params.xs_ggH.get_value(),
                params.br_hgg.get_value(),
                params.eff.get_value(),
                params.lumi.get_value(),
            ),
            name="signal_rate",
        )
        model_bkg = Exponential(
            lambd=params.lambd.get_value(), lower=mass.lower, upper=mass.upper
        )
        composed_mu = evm.Parameter(
            mean_function(
                params.higgs_mass.get_value(), params.d_higgs_mass.get_value()
            )
        )
        model_ggH = Gaussian(
            mu=composed_mu.get_value(),
            sigma=params.sigma,
            lower=mass.lower,
            upper=mass.upper,
        )
        model = SumPDF(
            pdfs=[model_ggH, model_bkg],
            lower=mass.lower,
            upper=mass.upper,
            extended_vals=[signal_rate, params.model_bkg_norm.get_value()],
        )
        nll = create_extended_nll(params, model, data)
        return jnp.squeeze(nll)

    graphdef, diffable, static = nnx.split(
        unwrap(params_card), evm.filter.is_dynamic_parameter, ...
    )

    def optx_loss_fn(diffable, args):
        return loss_fn_card(diffable, *args)

    fitresult = optimistix.minimise(
        optx_loss_fn,
        solver,
        diffable,
        has_aux=False,
        args=(static, data_sides, graphdef),
        options={},
        max_steps=1000,
        throw=True,
    )
    fitted_params = wrap(nnx.merge(graphdef, fitresult.value, static, copy=True))

    denominator = loss_fn_card(fitresult.value, static, data, graphdef)

    # === Final Results ===
    print(f"Final estimate: r = {fitted_params.r.get_value()}\n")

    # === Scan for r ===
    print("Scanning for r")

    # based on https://github.com/pfackeldey/evermore/blob/main/examples/nll_profiling.py
    def fixed_mu_fit(mu, params_card, silent=True):
        # params_card = eqx.tree_at(lambda t: t.r.get_value(), params_card, mu)
        # params_card = eqx.tree_at(lambda t: t.r.frozen, params_card, True)

        # Set & freeze the `mu` parameter in the model (it's mutable!):
        params_card.r[...] = mu
        params_card.r.set_metadata(frozen=True)

        # split the model into metadata (graphdef), dynamic and static part
        graphdef, diffable, static = nnx.split(
            unwrap(params_card), evm.filter.is_dynamic_parameter, ...
        )

        def optx_loss_fn(diffable, args):
            return loss_fn_card(diffable, *args)

        fitresult = optimistix.minimise(
            optx_loss_fn,
            solver,
            diffable,
            has_aux=False,
            args=(static, data_sides, graphdef),
            options={},
            max_steps=1000,
            throw=True,
        )
        fitted_params = wrap(nnx.merge(graphdef, fitresult.value, static, copy=True))
        loss = loss_fn_card(fitresult.value, static, data_sides, graphdef)

        if not silent:
            print(
                f"mu = {mu}, loss = {loss:.4f}, lambd = {fitted_params.lambd.get_value().astype(float)}, bkg_norm = {fitted_params.model_bkg_norm.get_value().astype(float)}"
            )
        return 2 * (loss - denominator)

    mus = jnp.linspace(0.95, 2.15, 20)
    two_nlls = []
    for mu in mus:
        two_nll = fixed_mu_fit(mu, params_card, silent=True)
        print(f"|> {mu=:.2f} -> {two_nll=:.2f}")
        two_nlls.append(two_nll)

    # vectorized version
    # two_nlls = jax.vmap(fixed_mu_fit)(mus)

    # plot
    print("\nPlotting scan")
    y = jnp.asarray(two_nlls)
    x = jnp.array(mus)

    func = interpolate.interp1d(x, y, kind="cubic")
    n_interp = 1000
    x_interp = np.linspace(x[0], x[-1], n_interp)
    y_interp = func(x_interp)
    y_interp = y_interp - np.min(y_interp)
    fig, ax = plt.subplots()
    ax.plot(x_interp, y_interp, label="scan", color="black")
    ax.set_xlabel("r")
    ax.set_ylabel("-2 ln L")
    # horizontal line at 1
    ax.axhline(y=1, color="r", linestyle="--", label="1 sigma")
    ax.set_ylim(0, 10)
    save_image("part2_scan", output_dir)

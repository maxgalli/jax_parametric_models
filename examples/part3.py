import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
from scipy import interpolate
import os
import jax.numpy as jnp
import evermore as evm
import equinox as eqx
from typing import NamedTuple, List
import jax
import optax
from pathlib import Path

from paramore import (
    EVMExponential,
    EVMGaussian,
    ExtendedNLL,
    GaussianConstraint,
    plot_as_data,
    save_image,
)


# double precision
jax.config.update("jax_enable_x64", True)

# plot styling
hep.style.use("CMS")

if __name__ == "__main__":
    # Signal modelling
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "samples"
    fl = data_dir / "mc_part1.parquet"
    output_dir = "figures_part1"
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_parquet(fl)
    var_name = "CMS_hgg_mass"

    fig, ax = plt.subplots()
    ax = plot_as_data(df[var_name], nbins=100, ax=ax)
    ax.set_xlabel("$m_{\gamma\gamma}$ [GeV]")
    save_image("part1_signal_mass", output_dir)

    data = jax.numpy.array(df["CMS_hgg_mass"].values)
    true_mean = 125.0

    higgs_mass = evm.Parameter(
        value=125.0, name="higgs_mass", lower=120.0, upper=130.0, frozen=True
    )
    d_higgs_mass = evm.Parameter(value=0.0, name="dMH", lower=-1.0, upper=1.0)
    sigma = evm.Parameter(value=2.0, name="sigma", lower=1.0, upper=5.0)


    class Params(NamedTuple):
        higgs_mass: evm.Parameter
        d_higgs_mass: evm.Parameter
        sigma: evm.Parameter


    params = Params(higgs_mass, d_higgs_mass, sigma)


    def mean_function(higgs_mass, d_higgs_mass):
        return higgs_mass + d_higgs_mass


    class NLL(eqx.Module):
        model: eqx.Module
        data: jax.Array

        def __call__(self):
            return -jnp.sum(jnp.log(self.model(self.data) + 1e-8))


    # === Loss Function ===
    @jax.jit
    def loss_fn(diffable, static, data):
        # return negative_log_likelihood(model, data)
        params = eqx.combine(diffable, static)
        std = params.sigma
        composed_mu = evm.Parameter(
            mean_function(params.higgs_mass.value, params.d_higgs_mass.value)
        )
        model = EVMGaussian(composed_mu, std)
        nll = NLL(model, data)
        return nll()


    # === Optimizer (adam) ===
    optimizer_settings = dict(learning_rate=3e-3, b1=0.999)
    optimizer = optax.adam(**optimizer_settings)
    opt_state = optimizer.init(eqx.filter(params, eqx.is_inexact_array))


    # === Training Step ===
    @jax.jit
    def step(params, opt_state, data):
        diffable, static = evm.tree.partition(params)
        loss, grads = jax.value_and_grad(loss_fn)(diffable, static, data)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = eqx.apply_updates(params, updates)
        return params, opt_state, loss


    # === Training Loop ===
    num_epochs = 1000
    for epoch in range(num_epochs):
        params, opt_state, loss = step(params, opt_state, data)
        if epoch % 100 == 0:
            print(f"{epoch=}: Loss = {loss:.4f}, Std = {params.sigma.value}")
            print(f"dMH = {params.d_higgs_mass.value}")

    # === Final Results ===
    print(f"Final estimate: Std = {params.sigma.value}")
    print(f"Final estimate: dMH = {params.d_higgs_mass.value}")

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
    var_name = "CMS_hgg_mass"
    # df_data_sides = df_data[(df_data[var_name] > 100) & (df_data[var_name] < 115) | (df_data[var_name] > 135) & (df_data[var_name] < 180)]
    # keep only data between 100 and 180
    df_data = df_data[(df_data[var_name] > 100) & (df_data[var_name] < 180)]
    # data_sides = jax.numpy.array(df_data_sides[var_name].values)
    data_sides = jax.numpy.array(df_data[var_name].values)
    lam = evm.Parameter(value=0.05, name="lambda", lower=0, upper=0.2)

    class ParamsBkg(NamedTuple):
        lambd: evm.Parameter

    params_bkg = ParamsBkg(lam)

    @jax.jit
    def loss_fn_bkg(diffable, static, data):
        params = eqx.combine(diffable, static)
        model = EVMExponential(params.lambd)
        nll = NLL(model, data)
        return nll()


    # === Optimizer (adam) ===
    optimizer = optax.adam(**optimizer_settings)
    opt_state = optimizer.init(eqx.filter(params_bkg, eqx.is_inexact_array))


    # === Training Step ===
    @jax.jit
    def step_bkg(params, opt_state, data):
        diffable, static = evm.tree.partition(params)
        loss, grads = jax.value_and_grad(loss_fn_bkg)(diffable, static, data)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = eqx.apply_updates(params, updates)
        return params, opt_state, loss


    # === Training Loop ===
    for epoch in range(num_epochs):
        params_bkg, opt_state, loss = step_bkg(params_bkg, opt_state, data_sides)
        if epoch % 100 == 0:
            print(f"{epoch=}: Loss = {loss:.4f}, Lambda = {params_bkg.lambd.value}")

    # === Final Results ===
    print(f"Final estimate: Lambda = {params_bkg.lambd.value}")

    fig, ax = plt.subplots()
    ax.set_xlabel("$m_{\gamma\gamma}$ [GeV]")
    x = np.linspace(100, 180, 1000)
    model_bkg = EVMExponential(params_bkg.lambd)
    y = model_bkg(x)
    ax.plot(x, y, label="fit")
    ax.legend()
    save_image("part1_data_sidebands", output_dir)

    ###
    # systematics
    ###
    output_dir = base_dir / "figures_part3_zfit"
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = Path(__file__).resolve().parent / "samples"
    nominal_file = samples_dir / "mc_part3_ggH_Tag0.parquet"
    file_template = "mc_part3_ggH_Tag0_{}01Sigma.parquet"
    var_name = "CMS_hgg_mass"

    # get nominal
    df_nominal = pd.read_parquet(nominal_file)

    # photonID has weights
    yield_variations = {}
    #for sys in ["JEC", "photonID"]:
    for sys in ["photonID"]:
        for direction in ["Up", "Down"]:
            fl = samples_dir / file_template.format(sys + direction)
            df = pd.read_parquet(fl)
            numerator = df[var_name] * df["weight"]
            denominator = df_nominal[var_name] * df_nominal["weight"]
            yld = numerator.sum() / denominator.sum()
            yld = yld.astype(np.float64)
            print("Systematic varied yield ({}, {}) = {:.3f}".format(sys, direction, yld))
            yield_variations[sys + direction] = yld

    #def yield_multiplicative_factor_asymm_function(theta):
    #    kappa_up = yield_variations["photonIDUp"]
    #    kappa_down = yield_variations["photonIDDown"]
    #    # see CAT-23-001-paper-v19.pdf pag 7
    #    if theta < -0.5:
    #        return kappa_down ** (-theta)
    #    elif theta > 0.5:
    #        return kappa_up**theta
    #    else:
    #        return jnp.exp(
    #            theta
    #            * (
    #                4 * jnp.log(kappa_up / kappa_down)
    #                + jnp.log(kappa_up * kappa_down)
    #                * (48 * theta**5 - 40 * theta**3 + 15 * theta)
    #            )
    #            / 8
    #        )
    #yield_multiplicative_factor_asymm_function = jax.jit(
    #    yield_multiplicative_factor_asymm_function,
    #    static_argnames="theta",
    #)

    def yield_multiplicative_factor_symm_function(theta):
        print("inside yield_multiplicative_factor_symm_function")
        print(f"theta = {theta}")
        return yield_variations["photonIDUp"] ** theta

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
    r = evm.Parameter(value=1.0, name="r", lower=0.0, upper=20.0)

    theta = evm.Parameter(value=0.0, name="theta", lower=-5.0, upper=5.0)

    def model_ggH_Tag0_norm_function(r, xs_ggH, br_hgg, eff, lumi, theta):
        return r * xs_ggH * br_hgg * eff * lumi * theta

    norm_bkg = evm.Parameter(
        value=float(df_data.shape[0]),
        name="model_bkg_Tag0_norm",
        lower=0.0,
        upper=float(3 * df_data.shape[0]),
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
        theta: evm.Parameter

    # redefine sigma and d_higgs_mass with the best value from before such that they are now frozen
    sigma = evm.Parameter(
        value=params.sigma.value, name="sigma", lower=1.0, upper=5.0, frozen=True
    )
    d_higgs_mass = evm.Parameter(
        value=params.d_higgs_mass.value, name="dMH", lower=-1.0, upper=1.0, frozen=True
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
        theta,
    )

    # === Loss Function ===
    @jax.jit
    def loss_fn_card(diffable, static, data):
        params = eqx.combine(diffable, static)
        th = yield_multiplicative_factor_symm_function(params.theta.value)
        print("inside loss_fn_card")
        print(f"theta = {params.theta.value}")
        signal_rate = evm.Parameter(
            model_ggH_Tag0_norm_function(
                params.r.value,
                params.xs_ggH.value,
                params.br_hgg.value,
                params.eff.value,
                params.lumi.value,
                th
            ),
            name="signal_rate",
        )
        model_bkg = EVMExponential(params.lambd)
        composed_mu = evm.Parameter(
            mean_function(params.higgs_mass.value, params.d_higgs_mass.value)
        )
        model_ggH = EVMGaussian(composed_mu, params.sigma)

        theta_constraint = GaussianConstraint(
            params.theta, 0.0, 1.0
        )
        constraints = [theta_constraint]
        
        nll = ExtendedNLL([model_bkg, model_ggH], [params.model_bkg_norm, signal_rate], constraints)
        return nll(data)

    # === Optimizer (adam) ===
    optimizer = optax.adam(**optimizer_settings)
    opt_state = optimizer.init(eqx.filter(params_card, eqx.is_inexact_array))


    # === Training Step ===
    @jax.jit
    def step_card(params, opt_state, data):
        diffable, static = evm.tree.partition(params)
        loss, grads = jax.value_and_grad(loss_fn_card)(diffable, static, data)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = eqx.apply_updates(params, updates)
        return params, opt_state, loss


    # === Training Loop ===
    num_epochs = 50000
    for epoch in range(num_epochs):
        params_card, opt_state, loss = step_card(params_card, opt_state, data_sides)
        if epoch % 100 == 0:
            print(f"{epoch=}: Loss = {loss:.4f}, r = {params_card.r.value}")
            print(f"alpha = {params_card.lambd.value}")
            print(f"Bkg = {params_card.model_bkg_norm.value}")
            print(f"theta = {params_card.theta.value}")

    denominator = loss

    # === Final Results ===
    print(f"Final estimate: r = {params_card.r.value}\n")

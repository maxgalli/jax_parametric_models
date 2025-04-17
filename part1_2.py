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
from jax import lax
import optax

from utils import plot_as_data
from utils import save_image

from distributions import EVMExponential, EVMGaussian, ExtendedNLL, GaussianConstraint

# double precision
jax.config.update("jax_enable_x64", True)

# plot styling
hep.style.use("CMS")

if __name__ == "__main__":
    # Signal modelling
    data_dir = "../StatsStudies/ExercisesForCourse/Hgg_zfit/data"
    fl = os.path.join(data_dir, "mc_part1.parquet")
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
        diffable, static = evm.parameter.partition(params)
        loss, grads = jax.value_and_grad(loss_fn)(diffable, static, data)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = eqx.apply_updates(params, updates)
        return params, opt_state, loss


    # === Training Loop ===
    num_epochs = 2000
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

    fl_data = os.path.join(data_dir, "data_part1.parquet")
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
        diffable, static = evm.parameter.partition(params)
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
    r = evm.Parameter(value=1.0, name="r", lower=0.0, upper=20.0)


    def model_ggH_Tag0_norm_function(r, xs_ggH, br_hgg, eff, lumi):
        return r * xs_ggH * br_hgg * eff * lumi


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
    )

    # === Loss Function ===
    @jax.jit
    def loss_fn_card(diffable, static, data):
        params = eqx.combine(diffable, static)
        signal_rate = evm.Parameter(
            model_ggH_Tag0_norm_function(
                params.r.value,
                params.xs_ggH.value,
                params.br_hgg.value,
                params.eff.value,
                params.lumi.value,
            ),
            name="signal_rate",
        )
        model_bkg = EVMExponential(params.lambd)
        composed_mu = evm.Parameter(
            mean_function(params.higgs_mass.value, params.d_higgs_mass.value)
        )
        model_ggH = EVMGaussian(composed_mu, params.sigma)
        nll = ExtendedNLL([model_bkg, model_ggH], [params.model_bkg_norm, signal_rate])
        return nll(data)


    # === Optimizer (adam) ===
    optimizer = optax.adam(**optimizer_settings)
    opt_state = optimizer.init(eqx.filter(params_card, eqx.is_inexact_array))


    # === Training Step ===
    @jax.jit
    def step_card(params, opt_state, data):
        diffable, static = evm.parameter.partition(params)
        loss, grads = jax.value_and_grad(loss_fn_card)(diffable, static, data)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = eqx.apply_updates(params, updates)
        return params, opt_state, loss


    # === Training Loop ===
    for epoch in range(num_epochs):
        params_card, opt_state, loss = step_card(params_card, opt_state, data_sides)
        if epoch % 100 == 0:
            print(f"{epoch=}: Loss = {loss:.4f}, r = {params_card.r.value}")
            print(f"alpha = {params_card.lambd.value}")
            print(f"Bkg = {params_card.model_bkg_norm.value}")

    denominator = loss

    # === Final Results ===
    print(f"Final estimate: r = {params_card.r.value}\n")


    # === Scan for r ===
    print("Scanning for r")


    # based on https://github.com/pfackeldey/evermore/blob/main/examples/nll_profiling.py
    def fixed_mu_fit(mu, silent=True):
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
            value=mu.astype(float), name="r", lower=0.0, upper=20.0, frozen=True
        )

        def model_ggH_Tag0_norm_function(r, xs_ggH, br_hgg, eff, lumi):
            return r * xs_ggH * br_hgg * eff * lumi

        norm_bkg = evm.Parameter(
            value=params_card.model_bkg_norm.value,
            name="model_bkg_Tag0_norm",
            lower=0.0,
            upper=float(3 * df_data.shape[0]),
        )
        lambd = evm.Parameter(
            value=params_card.lambd.value, name="lambda", lower=0.0, upper=0.2
        )

        # redefine sigma and d_higgs_mass with the best value from before such that they are now frozen
        sigma = evm.Parameter(
            value=params.sigma.value, name="sigma", lower=1.0, upper=5.0, frozen=True
        )
        d_higgs_mass = evm.Parameter(
            value=params.d_higgs_mass.value, name="dMH", lower=-1.0, upper=1.0, frozen=True
        )

        params_card_inside = ParamsCard(
            higgs_mass,
            d_higgs_mass,
            sigma,
            lambd,
            r,
            xs_ggH_par,
            br_hgg_par,
            eff_par,
            lumi_par,
            norm_bkg,
        )

        @jax.jit
        def loss_fn_card_inside(diffable, static, data):
            params = eqx.combine(diffable, static)
            # total_rate_par = evm.Parameter(total_rate(params.r.value, params.xs_ggH.value, params.br_hgg.value, params.eff.value, params.lumi.value, params.model_bkg_norm.value), name="total_rate")
            signal_rate = evm.Parameter(
                model_ggH_Tag0_norm_function(
                    params.r.value,
                    params.xs_ggH.value,
                    params.br_hgg.value,
                    params.eff.value,
                    params.lumi.value,
                ),
                name="signal_rate",
            )
            model_bkg = EVMExponential(params.lambd)
            composed_mu = evm.Parameter(
                mean_function(params.higgs_mass.value, params.d_higgs_mass.value)
            )
            model_ggH = EVMGaussian(composed_mu, params.sigma)
            nll = ExtendedNLL([model_bkg, model_ggH], [params.model_bkg_norm, signal_rate])
            return nll(data)

        optimizer = optax.adam(**optimizer_settings)
        opt_state = optimizer.init(eqx.filter(params_card_inside, eqx.is_inexact_array))

        @jax.jit
        def step_card_inside(params, opt_state, data):
            diffable, static = evm.parameter.partition(params)
            # loss, grads = jax.value_and_grad(loss_fn_card)(diffable, static, data)
            grads = eqx.filter_grad(loss_fn_card_inside)(diffable, static, data)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = eqx.apply_updates(params, updates)
            return params, opt_state

        # minimize model with 1000 steps
        if not silent:
            print(f"\nr = {mu}")

        for epoch in range(num_epochs):
            model, opt_state = step_card_inside(params_card_inside, opt_state, data_sides)
            if epoch % 100 == 0 and not silent:
                print(f"{epoch=}: alpha = {model.lambd.value}")
        dynamic_model, static_model = evm.parameter.partition(model)
        loss = loss_fn_card_inside(dynamic_model, static_model, data_sides)
        if not silent:
            print(
                f"mu = {mu}, loss = {loss:.4f}, lambd = {dynamic_model.lambd.value.astype(float)}, bkg_norm = {dynamic_model.model_bkg_norm.value.astype(float)}"
            )
        return 2 * (loss - denominator)


    mus = jnp.linspace(0.7, 2.1, 15)
    two_nlls = []
    for mu in mus:
        two_nll = fixed_mu_fit(mu, silent=True)
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
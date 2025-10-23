import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
from scipy import interpolate
from pathlib import Path
import jax.numpy as jnp
import evermore as evm
import equinox as eqx
from typing import NamedTuple, List
import jax
import optax
from copy import deepcopy
from dask.distributed import Client

from paramore import (
    EVMExponential,
    EVMGaussian,
    EVMSumPDF,
    ExtendedNLL,
    plot_as_data,
    save_image,
)
from evermore.parameters.transform import MinuitTransform, unwrap, wrap
import optimistix

# double precision
jax.config.update("jax_enable_x64", True)

# plot styling
hep.style.use("CMS")

def main():
    # gauss for signal
    true_mean = 125.0

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
    d_higgs_mass = evm.Parameter(value=0.0, name="dMH", lower=-1.0, upper=1.0, frozen=True)
    sigma = evm.Parameter(value=2.0, name="sigma", lower=1.0, upper=5.0, frozen=True)

    def mean_function(higgs_mass, d_higgs_mass):
        return higgs_mass + d_higgs_mass

    composed_mu = evm.Parameter(
        mean_function(higgs_mass.value, d_higgs_mass.value)
    )

    # norm signal
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    fl = data_dir / "mc_part1.parquet"
    df = pd.read_parquet(fl)
    xs_ggH = 48.58  # pb
    br_hgg = 0.0027
    sumw = df["weight"].sum()
    eff = sumw / (xs_ggH * br_hgg)
    lumi = 138000.0

    minuit_transform = MinuitTransform()

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
    r = evm.Parameter(value=0.5, name="r", lower=0.0, upper=20.0, transform=minuit_transform)

    def model_ggH_Tag0_norm_function(r, xs_ggH, br_hgg, eff, lumi):
        return r * xs_ggH * br_hgg * eff * lumi

    signal_rate = evm.Parameter(
        model_ggH_Tag0_norm_function(
            r.value,
            xs_ggH_par.value,
            br_hgg_par.value,
            eff_par.value,
            lumi_par.value,
        ),
        name="signal_rate",
    )
    print(signal_rate.value)

    # signal
    gauss = EVMGaussian(var=mass, mu=composed_mu, sigma=sigma, extended=signal_rate)


    # background
    lam = evm.Parameter(value=0.05, name="lambda", lower=0, upper=0.2, frozen=True)

    # background rate
    fl_data = data_dir / "data_part1.parquet"
    df_data = pd.read_parquet(fl_data)

    norm_bkg = evm.Parameter(
        value=float(df_data.shape[0]),
        name="model_bkg_Tag0_norm",
        lower=0.0,
        upper=float(3 * df_data.shape[0]),
        frozen=False,
        transform=minuit_transform,
    )
    print(norm_bkg.value)
    bkg = EVMExponential(var=mass,lambd=lam, extended=norm_bkg)

    # full model
    model = EVMSumPDF(
        var=mass,
        pdfs=[gauss, bkg],
    )

    #nevents = len(df_data)
    # nevents = 10181
    # print("number of events:", nevents)
    ntoys = 20
    key = jax.random.PRNGKey(0)

    toy = jax.vmap(model.sample)(jax.random.split(key, ntoys))
    # toy = [
    #     model.sample(
    #         key=key,
    #         #sample_shape=(nevents,),
    #         # xmin=df_data["CMS_hgg_mass"].min(),
    #         # xmax=df_data["CMS_hgg_mass"].max(),
    #         #xmax=250.0,  # max mass
    #     ) for key in jax.random.split(key, ntoys)
    # ]
    # concatenate
    #toy = toy[0]  # take the first toy
    # plot the toy
    fig, ax = plt.subplots()
    ax.hist(
        toy[:10],
        bins=50,
        histtype="step",
        #label="Toy data",
        #alpha=0.5,
        #color="blue",
        #density=True,
    )
    ax.set_ylabel("Events")
    output_dir = base_dir / "figures_part1"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_image("toy_data", output_dir)

    # best fit on toys
    class ParamsCard(NamedTuple):
    #class ParamsCard(eqx.Module):
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

    def make_params_card():
        return ParamsCard(
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
    def loss_fn_card(diffable, static, data):
        params = wrap(evm.tree.combine(diffable, static))
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
        model_bkg = EVMExponential(var=mass,lambd=params.lambd, extended=params.model_bkg_norm)
        composed_mu = evm.Parameter(
            mean_function(params.higgs_mass.value, params.d_higgs_mass.value)
        )
        model_ggH = EVMGaussian(var=mass,mu=composed_mu, sigma=params.sigma, extended=signal_rate)
        model= EVMSumPDF(var=mass, pdfs=[model_bkg, model_ggH])
        #nll = ExtendedNLL([model_bkg, model_ggH], [params.model_bkg_norm, signal_rate])
        nll = ExtendedNLL(model)
        return nll(data)


    diffable, static = evm.tree.partition(unwrap(make_params_card()))

    def optx_loss_fn(diffable, args):
        return loss_fn_card(diffable, *args)

    solver = optimistix.BFGS(
        rtol=1e-5, atol=1e-7, verbose=frozenset({"step_size", "loss"})
    )

    from IPython import embed; embed()
    
    params_after = []
    for t in toy:
        fitresult = optimistix.minimise(
            optx_loss_fn,
            solver,
            diffable,
            has_aux=False,
            args=(static, t),
            options={},
            max_steps=1000,
            throw=True,
        )
        fitted_params = wrap(evm.tree.combine(fitresult.value, static))
        params_after.append(fitted_params)

    #for i, t in enumerate(toys):
    #    np = train(t, params_card_list[i], opt_state_list[i])
    #    print(f"From toy {i}:")
    #    print(f"r = {np.r.value[0]:.4f}")
    #    params_after.append(np)
    #    #params_after.append(train(t, params_card_list[i], opt_state_list[i]))

    #params_after = jax.vmap(train, in_axes=(0, 0, 0))(
    #    jnp.array(toys),
    #    jnp.array(params_card_list),
    #    jnp.array(opt_state_list)
    #    )
    
    dist_r = jnp.array([p.r.value for p in params_after])
    mean_r = jnp.mean(dist_r)
    std_r = jnp.std(dist_r)
    print(f"mean r: {mean_r}, std r: {std_r}")
    #for p in params_after:
    #    print(p.r.value)

if __name__ == "__main__":
    main()

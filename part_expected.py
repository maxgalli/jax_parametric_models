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
from copy import deepcopy
from dask.distributed import Client

from utils import plot_as_data
from utils import save_image

from distributions import EVMExponential, EVMGaussian, EVMMixture, ExtendedNLL

# double precision
jax.config.update("jax_enable_x64", True)

# plot styling
hep.style.use("CMS")

def main():
    # gauss for signal
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

    gauss = EVMGaussian(composed_mu, sigma)

    # norm signal
    data_dir = "../StatsStudies/ExercisesForCourse/Hgg_zfit/data"
    fl = os.path.join(data_dir, "mc_part1.parquet")
    df = pd.read_parquet(fl)
    xs_ggH = 48.58  # pb
    br_hgg = 0.0027
    sumw = df["weight"].sum()
    eff = sumw / (xs_ggH * br_hgg)
    lumi = 138000.0

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
    r = evm.Parameter(value=0.5, name="r", lower=0.0, upper=20.0)

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

    # background
    lam = evm.Parameter(value=0.05, name="lambda", lower=0, upper=0.2, frozen=True)
    bkg = EVMExponential(lam)

    # background rate
    fl_data = os.path.join(data_dir, "data_part1.parquet")
    df_data = pd.read_parquet(fl_data)

    norm_bkg = evm.Parameter(
        value=float(df_data.shape[0]),
        name="model_bkg_Tag0_norm",
        lower=0.0,
        upper=float(3 * df_data.shape[0]),
        frozen=False,
    )
    print(norm_bkg.value)
    # full model
    model = EVMMixture(
        components=[gauss, bkg],
        weights=[signal_rate, norm_bkg],
    )

    #nevents = len(df_data)
    nevents = 10181
    print("number of events:", nevents)
    ntoys = 100
    key = jax.random.PRNGKey(0)
    toy_list = []
    for i in range(1, 10):
        toy = model.sample(
            seed=key,
            #sample_shape=(nevents,),
            sample_shape=(ntoys, nevents), # (number_of_toys, nevents)
            xmin=df_data["CMS_hgg_mass"].min(),
            xmax=df_data["CMS_hgg_mass"].max(), 
            #xmax=250.0,  # max mass
        )
        toy_list.append(toy)
    # concatenate
    toy = jnp.concatenate(toy_list, axis=0)
    #toy = toy[0]  # take the first toy
    toys = [t for t in toy]  # list of toys
    # plot the toy
    fig, ax = plt.subplots()
    ax.hist(
        toys[:10],
        bins=50,
        histtype="step",
        #label="Toy data",
        #alpha=0.5,
        #color="blue",
        #density=True,
    )
    ax.set_ylabel("Events")
    output_dir = "figures_part1"
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
        #nll = ExtendedNLL([model_bkg, model_ggH], [params.model_bkg_norm, signal_rate])
        nll = ExtendedNLL([model_ggH, model_bkg], [signal_rate, params.model_bkg_norm])
        return nll(data)

    # === Optimizer (adam) ===
    optimizer_settings = dict(learning_rate=3e-3, b1=0.999)
    optimizer = optax.adam(**optimizer_settings)
    def make_opt_state(params):
        return optimizer.init(eqx.filter(params, eqx.is_inexact_array))
    #opt_state = optimizer.init(eqx.filter(params_card, eqx.is_inexact_array))

    num_epochs = 1000

    # === Training Step ===
    @jax.jit
    def step_card(params, opt_state, data):
        diffable, static = evm.parameter.partition(params)
        loss, grads = jax.value_and_grad(loss_fn_card)(diffable, static, data)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = eqx.apply_updates(params, updates)
        return params, opt_state, loss
    
    def train(data, params, opt_state):
        for epoch in range(num_epochs):
            params, opt_state, loss = step_card(
                params, opt_state, data
            )
            if epoch % 100 == 0:
                r_val = params.r.value[0]
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
                #print(f"r = {r_val}")
                print(f"r = {r_val:.4f}")
                #print(f"lambd = {params_card.lambd.value:.4f}")
        return params

    # === Training ===
    params_card_list = []
    opt_state_list = []
    params_after = []
    for t in toys:
        #params_card_list.append(deepcopy(params_card))
        #opt_state_list.append(deepcopy(opt_state))
        params_card = make_params_card()
        opt_state = make_opt_state(params_card)
        params_card_list.append(params_card)
        opt_state_list.append(opt_state)

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
    
    client = Client()
    params_after = client.map(
        train,
        toys,
        params_card_list,
        opt_state_list,
    )
    params_after = client.gather(params_after)
    
    dist_r = jnp.array([p.r.value for p in params_after])
    mean_r = jnp.mean(dist_r)
    std_r = jnp.std(dist_r)
    print(f"mean r: {mean_r}, std r: {std_r}")
    #for p in params_after:
    #    print(p.r.value)

if __name__ == "__main__":
    main()
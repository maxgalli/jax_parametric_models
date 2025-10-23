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
    ExtendedNLL,
    plot_as_data,
    save_image,
)

# double precision
jax.config.update("jax_enable_x64", True)

# plot styling
hep.style.use("CMS")

def main():
    # gauss for signal
    higgs_mass = evm.Parameter(
        value=125.0, name="higgs_mass", lower=120.0, upper=130.0, frozen=False
    )
    d_higgs_mass = evm.Parameter(value=0.0, name="dMH", lower=-1.0, upper=1.0, frozen=True)
    sigma = evm.Parameter(value=2.0, name="sigma", lower=1.0, upper=5.0, frozen=True)

    def mean_function(higgs_mass, d_higgs_mass):
        return higgs_mass + d_higgs_mass

    composed_mu = evm.Parameter(
        mean_function(higgs_mass.value, d_higgs_mass.value)
    )

    model = EVMGaussian(composed_mu, sigma)
    #model = EVMGaussian(higgs_mass, sigma)

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

    # background rate
    fl_data = data_dir / "data_part1.parquet"
    df_data = pd.read_parquet(fl_data)

    norm_bkg = evm.Parameter(
        value=float(df_data.shape[0]),
        name="model_bkg_Tag0_norm",
        lower=0.0,
        upper=float(3 * df_data.shape[0]),
        frozen=True,
    )

    nevents = len(df_data)
    ntoys = 10
    key = jax.random.PRNGKey(0)
    toy_list = []
    for i in range(1, 30):
        print(i)
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
    output_dir = base_dir / "figures_part1"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_image("toy_data", output_dir)

    # best fit on toys
    class ParamsCard(NamedTuple):
    #class ParamsCard(eqx.Module):
        higgs_mass: evm.Parameter
        d_higgs_mass: evm.Parameter
        sigma: evm.Parameter

    def make_params_card():
        return ParamsCard(
            higgs_mass,
            d_higgs_mass,
            sigma,
        )

    # === Loss Function ===
    @jax.jit
    def loss_fn_card(diffable, static, data):
        params = eqx.combine(diffable, static)
        composed_mu = evm.Parameter(
            mean_function(params.higgs_mass.value, params.d_higgs_mass.value)
        )
        model_ggH = EVMGaussian(composed_mu, params.sigma)
        #nll = ExtendedNLL([model_bkg, model_ggH], [params.model_bkg_norm, signal_rate])
        #return nll(data)
        return -jnp.sum(model_ggH.log_prob(data))

    # === Optimizer (adam) ===
    optimizer_settings = dict(learning_rate=3e-3, b1=0.999)
    optimizer = optax.adam(**optimizer_settings)
    def make_opt_state(params):
        return optimizer.init(eqx.filter(params, eqx.is_inexact_array))
    #opt_state = optimizer.init(eqx.filter(params_card, eqx.is_inexact_array))

    num_epochs = 500

    # === Training Step ===
    @jax.jit
    def step_card(params, opt_state, data):
        diffable, static = evm.tree.partition(params)
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
                #r_val = params.r.value[0]
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
                ##print(f"r = {r_val}")
                #print(f"r = {r_val:.4f}")
                ##print(f"lambd = {params_card.lambd.value:.4f}")
                #pass
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
    print(params_after)
    
    dist_r = jnp.array([p.higgs_mass.value for p in params_after])
    mean_r = jnp.mean(dist_r)
    std_r = jnp.std(dist_r)
    print(f"mean mass: {mean_r}, std mass: {std_r}")
    #for p in params_after:
    #    print(p.higgs_mass.value)

#from oryx.distributions import Normal
#
#def main():
#    model = Normal(
#        loc=3.0,
#        scale=2.0,
#    )
#    # Generate toy data
#    nevents = 1000
#    toys = []
#    key = jax.random.PRNGKey(0)
#    for i in range(10):
#        key, subkey = jax.random.split(key)
#        toy = model.sample(seed=subkey, sample_shape=(nevents,))
#        toys.append(toy)
#    
#    # fit the model to the toy data
#    class Params(NamedTuple):
#        mu: jnp.ndarray
#        sigma: jnp.ndarray
#        
#    # create the parameters
#    params = Params(mu=jnp.array(3.0), sigma=jnp.array(2.0))
#
#    # loss
#    @jax.jit
#    def loss_fn(params, data):
#        #params = eqx.combine(diffable, static)
#        #model = Normal(loc=params.mu, scale=params.sigma)
#        #nll = -jnp.sum(model.log_prob(data))
#        model = Normal(loc=params.mu, scale=params.sigma)
#        nll = -jnp.sum(model.log_prob(data))
#        return nll
#
#    # optimizer
#    optimizer_settings = dict(learning_rate=3e-3, b1=0.999)
#    optimizer = optax.adam(**optimizer_settings)
#    opt_state = optimizer.init(eqx.filter(params, eqx.is_inexact_array)) 
#
#    # training step
#    @jax.jit
#    def step(params, opt_state, data):
#        #diffable, static = evm.tree.partition(params)
#        loss, grads = jax.value_and_grad(loss_fn)(params, data)
#        updates, opt_state = optimizer.update(grads, opt_state)
#        params = eqx.apply_updates(params, updates)
#        return params, opt_state, loss
#    
#    # training
#    num_epochs = 100
#    #params_list = []
#    #opt_state_list = []
#    params_after = []
#    for t in toys:
#        params = Params(mu=jnp.array(3.0), sigma=jnp.array(2.0))
#        opt_state = optimizer.init(eqx.filter(params, eqx.is_inexact_array))
#        #params_list.append(params)
#        #opt_state_list.append(opt_state)
#        for epoch in range(num_epochs):
#            params, opt_state, loss = step(
#                params, opt_state, t
#            )
#            if epoch % 10 == 0:
#                print(f"Epoch {epoch}, Loss: {loss:.4f}")
#                print(f"mu = {params.mu:.4f}")
#                print(f"sigma = {params.sigma:.4f}")
#        params_after.append(params)
#    
#    # gather the results
#    dist_mu = jnp.array([p.mu for p in params_after])
#    mean_mu = jnp.mean(dist_mu)
#    std_mu = jnp.std(dist_mu)
#    print(f"mean mu: {mean_mu}, std mu: {std_mu}")
    

if __name__ == "__main__":
    main()

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

from distributions import EVMExponential, EVMGaussian, ExtendedNLL, GaussianConstraint, EVMSumPDF

jax.config.update("jax_enable_x64", True)

if __name__ == "__main__":
    df = pd.read_parquet("../debug.parquet")
    data = jax.numpy.array(df["x"].values)

    x = evm.Parameter(value=5., name="x", lower=0., upper=10., frozen=False)
    mean = evm.Parameter(value=5., name="mean", lower=0., upper=10., frozen=False)
    sigma = evm.Parameter(value=1., name="sigma", lower=0.1, upper=3., frozen=False)
    lam = evm.Parameter(value=1., name="lambda", lower=0., upper=0., frozen=False)
    ns = evm.Parameter(value=500., name="ns", lower=0., upper=10000., frozen=True)
    nb = evm.Parameter(value=1500., name="nb", lower=0., upper=10000., frozen=True)
    mu = evm.Parameter(value=0., name="mu", lower=-5., upper=5., frozen=False)

    class Params(NamedTuple):
        x: evm.Parameter
        mean: evm.Parameter
        sigma: evm.Parameter
        lam: evm.Parameter
        ns: evm.Parameter
        nb: evm.Parameter
        mu: evm.Parameter

    params = Params(x=x, mean=mean, sigma=sigma, lam=lam, ns=ns, nb=nb, mu=mu)

    def sig_rate(ns, mu):
        return ns * mu

    # === Loss Function ===
    @jax.jit
    def loss_fn_card(diffable, static, data):
        params = eqx.combine(diffable, static)
        signal_rate = evm.Parameter(
            sig_rate(
                params.ns.value,
                params.mu.value,
            ),
            name="signal_rate",
        )
        model_bkg = EVMExponential(var=x, lambd=params.lam, extended=params.nb)
        model_sig = EVMGaussian(var=x, mu=params.mean, sigma=params.sigma, extended=signal_rate)
        model = EVMSumPDF(var=x, pdfs=[model_sig, model_bkg])
        #nll = ExtendedNLL([model_bkg, model_ggH], [params.model_bkg_norm, signal_rate])
        nll = ExtendedNLL(model=model)
        return nll(data)

    # === Optimizer (adam) ===
    optimizer_settings = dict(learning_rate=3e-3, b1=0.999)
    optimizer = optax.adam(**optimizer_settings)
    opt_state = optimizer.init(eqx.filter(params, eqx.is_inexact_array))


    # === Training Step ===
    @jax.jit
    def step_card(params, opt_state, data):
        diffable, static = evm.parameter.partition(params)
        loss, grads = jax.value_and_grad(loss_fn_card)(diffable, static, data)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = eqx.apply_updates(params, updates)
        return params, opt_state, loss


    # === Training Loop ===
    num_epochs = 10000
    for epoch in range(num_epochs):
        #params_card, opt_state, loss = step_card(params_card, opt_state, data_sides)
        params, opt_state, loss = step_card(params, opt_state, data)
        if epoch % 100 == 0:
            print(f"{epoch=}: Loss = {loss:.4f}, r = {params.mu.value}")
            print(f"alpha = {params.lam.value}")
            print(f"Bkg = {params.nb.value}")

    denominator = loss

    # === Final Results ===
    print(f"Final estimate: r = {params.mu.value}\n")
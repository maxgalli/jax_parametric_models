import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from iminuit import cost, Minuit
import mplhep as hep
from scipy.integrate import quad
from scipy import interpolate
import os
import jax.numpy as jnp
import evermore as evm
import equinox as eqx
from typing import NamedTuple
import jax
import optax

from jax import jacrev, jacfwd

from utils import plot_as_data
from utils import save_image

hep.style.use("CMS")

#class NLL(eqx.Module):
#    model: eqx.Module
#    data: jax.Array
#
#    def __call__(self):
#        #return -jnp.sum(jnp.log(self.model(self.data) + 1e-8))
#        return jnp.mean(-jnp.sum(jnp.log(self.model(self.data) + 1e-8)))
#
#class Gaussian:
#    def __init__(self, mu, sigma):
#        self.mu = mu
#        self.sigma = sigma
#    
#    def __call__(self, x):
#        return (1 / (jnp.sqrt(2 * jnp.pi * self.sigma.value**2))) * jnp.exp(-0.5 * ((x - self.mu.value) / self.sigma.value)**2)
#        #return jnp.exp(-(((x - self.mu.value) / self.sigma.value) ** 2) / 2)
#
#class Exponential:
#    def __init__(self, lambd):
#        self.lambd = lambd
#    
#    def __call__(self, x):
#        return self.lambd.value * jnp.exp(-self.lambd.value * x)
    
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

## === Base Model (Abstract) ===
#class BaseModel(eqx.Module):
#    def likelihood(self, data):
#        raise NotImplementedError
#
## === Gaussian Model (Fixed Mean, Trainable Std) ===
#class GaussianModel(BaseModel):
#    #mean: float = eqx.static_field()  # Fixed mean (not optimized)
#    #std: jax.Array  # Optimized parameter
#    mean: evm.Parameter
#    std: evm.Parameter
#
#    def __init__(self, mean, std):
#        #self.mean = true_mean  # Fixed mean
#        #self.std = jnp.array(2.0)  # Start with std = 1.0 (log space)
#        self.mean = mean
#        self.std = std
#
#    def likelihood(self, data):
#        """Computes Gaussian likelihood given data."""
#        #std = jnp.exp(self.log_std)  # Convert log_std to std
#        #return jnp.exp(-0.5 * ((data - self.mean) / std) ** 2) / (std * jnp.sqrt(2 * jnp.pi))
#        
#        #return jnp.exp(-0.5 * ((data - self.mean) / self.std) ** 2) / (self.std * jnp.sqrt(2 * jnp.pi))
#        return jnp.exp(-0.5 * ((data - self.mean.value) / self.std.value) ** 2) / (self.std.value * jnp.sqrt(2 * jnp.pi))
#
## === Generic Negative Log-Likelihood Function ===
#def negative_log_likelihood(model: BaseModel, data):
#    """Computes the negative log-likelihood for a given model."""
#    likelihoods = model.likelihood(data)
#    log_likelihood = jnp.sum(jnp.log(likelihoods + 1e-8))  # Add epsilon to avoid log(0)
#    return -log_likelihood  # Negative for minimization

class Gauss(eqx.Module):
    mean: evm.Parameter
    std: evm.Parameter

    def __call__(self, x):
        return jnp.exp(-0.5 * ((x - self.mean.value) / self.std.value) ** 2) / (self.std.value * jnp.sqrt(2 * jnp.pi))

class Expo(eqx.Module):
    lambd: evm.Parameter

    def __call__(self, x):
        return self.lambd.value * jnp.exp(-self.lambd.value * x)

class NLL(eqx.Module):
    model: eqx.Module
    data: jax.Array
    
    def __call__(self):
        return -jnp.sum(jnp.log(self.model(self.data) + 1e-8))

data = jax.numpy.array(df["CMS_hgg_mass"].values)
true_mean = 125.0

higgs_mass = evm.Parameter(value=125., name="higgs_mass", lower=120., upper=130., frozen=True)
d_higgs_mass = evm.Parameter(value=0., name="dMH", lower=-1., upper=1.)
sigma = evm.Parameter(value=2., name="sigma", lower=1., upper=5.)
class Params(NamedTuple):
    higgs_mass: evm.Parameter
    d_higgs_mass: evm.Parameter
    sigma: evm.Parameter
params = Params(higgs_mass, d_higgs_mass, sigma)

def mean_function(higgs_mass, d_higgs_mass):
    return higgs_mass + d_higgs_mass

# === Loss Function ===
@jax.jit
def loss_fn(diffable, static, data):
    #return negative_log_likelihood(model, data)
    params = eqx.combine(diffable, static)
    std = params.sigma
    composed_mu = evm.Parameter(mean_function(params.higgs_mass.value, params.d_higgs_mass.value))
    model = Gauss(composed_mu, std)
    nll = NLL(model, data)
    return nll()

# === Optimizer (Adam) ===
learning_rate = 0.05
optimizer = optax.adam(learning_rate)
#opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
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
num_epochs = 500
for epoch in range(num_epochs):
    params, opt_state, loss = step(params, opt_state, data)
    if epoch % 50 == 0:
        #print(f"Epoch {epoch}: Loss = {loss:.4f}, Std = {model.std:.4f}")
        print(f"Epoch {epoch}: Loss = {loss:.4f}, Std = {params.sigma.value}")
        print(f"dMH = {params.d_higgs_mass.value}")

# === Final Results ===
#print(f"Final estimate: Std = {model.std:.4f}")
print(f"Final estimate: Std = {params.sigma.value}")
print(f"Final estimate: dMH = {params.d_higgs_mass.value}")

# Signal normalisation
print("Getting signal normalisation")
xs_ggH = 48.58  # pb
br_hgg = 2.7e-3

# this part is slightly different from what jon does
sumw = df["weight"].sum()
eff = sumw / (xs_ggH * br_hgg)
print(f"Efficiency of ggH events landing in Tag0 is: {eff:.5f}")

lumi = 138000
n = eff * xs_ggH * br_hgg * lumi
print(f"For 138 fb^-1, the expected number of ggH events is: N = xs * BR * eff * lumi = {n:.5f}")

fl_data = os.path.join(data_dir, "data_part1.parquet")
df_data = pd.read_parquet(fl_data)
var_name = "CMS_hgg_mass"
df_data_sides = df_data[(df_data[var_name] > 100) & (df_data[var_name] < 115) | (df_data[var_name] > 135) & (df_data[var_name] < 180)]
#data_sides = jax.numpy.array(df_data_sides[var_name].values)
data_sides = jax.numpy.array(df_data[var_name].values)
lam = evm.Parameter(value=0.05, name="lambda", lower=0, upper=0.2)
class ParamsBkg(NamedTuple):
    lambd: evm.Parameter
params_bkg = ParamsBkg(lam)
model_bkg = Expo(lam)

@jax.jit
def loss_fn_bkg(diffable, static, data):
    params = eqx.combine(diffable, static)
    model = Expo(params.lambd)
    nll = NLL(model, data)
    return nll()

# === Optimizer (Adam) ===
learning_rate = 0.05
optimizer = optax.adam(learning_rate)
opt_state = optimizer.init(eqx.filter(params_bkg, eqx.is_inexact_array))

# === Training Step ===
@jax.jit
def step_bkg(params, opt_state, data):
    diffable, static = evm.parameter.partition(params)
    print(diffable)
    print(static)
    loss, grads = jax.value_and_grad(loss_fn_bkg)(diffable, static, data)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = eqx.apply_updates(params, updates)
    return params, opt_state, loss

# === Training Loop ===
num_epochs = 500
for epoch in range(num_epochs):
    params_bkg, opt_state, loss = step_bkg(params_bkg, opt_state, data_sides)
    if epoch % 50 == 0:
        print(f"Epoch {epoch}: Loss = {loss:.4f}, Lambda = {params_bkg.lambd.value}")

# === Final Results ===
print(f"Final estimate: Lambda = {params_bkg.lambd.value}")
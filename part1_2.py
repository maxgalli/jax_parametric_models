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

hep.style.use("CMS")

class NLL:
    def __init__(self, model, data):
        self.model = model
        self.data = data
    
    def __call__(self):
        return -jnp.sum(jnp.log(self.model(self.data)))

class Gaussian:
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma
    
    def __call__(self, x):
        return (1 / (jnp.sqrt(2 * jnp.pi * self.sigma.value**2))) * jnp.exp(-0.5 * ((x - self.mu.value) / self.sigma.value)**2)

class Exponential:
    def __init__(self, lambd):
        self.lambd = lambd
    
    def __call__(self, x):
        return self.lambd.value * jnp.exp(-self.lambd.value * x)
    
# Signal modelling
data_dir = "../StatsStudies/ExercisesForCourse/Hgg_zfit/data"
fl = os.path.join(data_dir, "mc_part1.parquet")
output_dir = "figures_part1"
os.makedirs(output_dir, exist_ok=True)
df = pd.read_parquet(fl)

higgs_mass = evm.Parameter(125, "higgs_mass", 120, 130, frozen=True)
sigma = evm.Parameter(2, "sigma", 1, 5)

def loss(higgs_mass, sigma):
    model = Gaussian(higgs_mass, sigma)
    nll = NLL(model, df["CMS_hgg_mass"].values)
    return nll()

grads = eqx.filter_grad(loss)(higgs_mass, sigma)
print(grads.sigma.value)

"""
#d_higgs_mass = evm.Parameter(0, "dMH", -1, 1)
#def mean_function(higgs_mass, d_higgs_mass):
#    return higgs_mass + d_higgs_mass
#mean = evm.ComposedParameter("mean", mean_function, (higgs_mass, d_higgs_mass))
#model = Gaussian(mean, sigma)
#nll = NLL(model, df["CMS_hgg_mass"].values)
#
#minimizer = Minuit(nll, *initial_values)
#result = minimizer.migrad()
#print(result)

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
lam = evm.Parameter(0.01, "lambda", 0, 0.1)
model_bkg = Exponential(lam)
nll = NLL(model_bkg, df_data_sides[var_name].values)
initial_values = [lam.value]
minimizer = Minuit(nll, *initial_values)
result = minimizer.migrad()
print(result)
"""
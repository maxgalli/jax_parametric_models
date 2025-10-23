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
from evermore.parameters.transform import MinuitTransform, unwrap, wrap
import optimistix


from paramore import (
    EVMExponential,
    EVMGaussian,
    ExtendedNLL,
    GaussianConstraint,
    EVMSumPDF,
    plot_as_data,
    save_image,
)


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
    data_dir = base_dir / "data"
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

    def model_ggH_Tag0_norm_function(r, xs_ggH, br_hgg, eff, lumi):
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
            0.0, name="d_higgs_mass", lower=-1.0, upper=1.0, transform=minuit_transform
        ),
        higgs_width=evm.Parameter(
            2.0, name="higgs_width", lower=1.0, upper=5.0, transform=minuit_transform
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

    def datacard(params):
        dc = {
            "only_channel": {
                "processes": {
                    "signal": {
                        "pdf": EVMGaussian(
                            var=mass,
                            mu=evm.Parameter(
                                mean_function(
                                    params.higgs_mass.value, params.d_higgs_mass.value
                                )
                            ),
                            sigma=params.higgs_width,
                            extended=evm.Parameter(
                                model_ggH_Tag0_norm_function(
                                    params.mu.value,
                                    xs_ggH,
                                    br_hgg,
                                    eff,
                                    lumi,
                                )
                            ),
                        ),
                        "expected_yield": None,
                        "nuisances": [
                            {
                                "name": "phoid_syst",
                                "type": "lnN",
                                "parameter": params.phoid_syst,
                                "value": 1.05,
                            }
                        ],
                    },
                    "background": {
                        "pdf": EVMExponential(
                            var=mass,
                            lambd=params.lamb,
                            extended=params.bkg_norm,
                        ),
                        "expected_yield": None,
                        "nuisances": [],
                    },
                },
                "observations": data,
            }
        }
        return dc

    def dc_to_nll(dc):
        nll_terms = []
        for channel_name, channel in dc.items():
            data = channel["observations"]
            pdfs = []
            constraints = []
            for proc_name, proc in channel["processes"].items():
                pdf = proc["pdf"]
                for nuisance_name in proc["nuisances"]:
                    if nuisance_name["type"] == "lnN":
                        print(pdf.extended)
                        new_extended = pdf.extended * jnp.exp(
                            nuisance_name["parameter"].value
                            * jnp.log(nuisance_name["value"])
                        )
                        pdf = eqx.tree_at(lambda p: p.extended, pdf, new_extended)
                        print(pdf.extended)
                    constraints.append(evm.loss.get_log_probs(nuisance_name["parameter"]))
                pdfs.append(pdf)
            total_pdf = EVMSumPDF(var=mass, pdfs=pdfs)
            nll = ExtendedNLL(model=total_pdf)
            nll_value = nll(data)
            if constraints:
                nll_value += evm.util.sum_over_leaves(constraints)
            nll_terms.append(nll_value)
        total_nll = jnp.sum(jnp.array(nll_terms))
        return total_nll
    
    print(dc_to_nll(datacard(params)))

    @eqx.filter_jit
    def loss(diffable, static, data):
        params = wrap(evm.tree.combine(diffable, static))
        dc = datacard(params)
        nll = dc_to_nll(dc)
        return nll

    diffable, static = evm.tree.partition(unwrap(params))

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

    print(f"Final estimate: r = {fitted_params.mu.value}\n")

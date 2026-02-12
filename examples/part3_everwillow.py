from pathlib import Path

import evermore as evm
import jax
import jax.numpy as jnp
import mplhep as hep
import optimistix
import pandas as pd
from evermore.parameters import filter as evm_filter
from evermore.parameters.transform import MinuitTransform, unwrap, wrap
from flax import nnx
from jax.experimental import checkify

import paramore as pm

wrap_checked = checkify.checkify(wrap)

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
    data_dir = base_dir / "samples"
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

    class Params(nnx.Pytree):
        def __init__(
            self,
            higgs_mass,
            d_higgs_mass,
            higgs_width,
            lamb,
            bkg_norm,
            mu,
            phoid_syst,
        ):
            self.higgs_mass = higgs_mass
            self.d_higgs_mass = d_higgs_mass
            self.higgs_width = higgs_width
            self.lamb = lamb
            self.bkg_norm = bkg_norm
            self.mu = mu
            self.phoid_syst = phoid_syst

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
                        "pdf": pm.Gaussian(
                            mu=mean_function(
                                params.higgs_mass.get_value(),
                                params.d_higgs_mass.get_value(),
                            ),
                            sigma=params.higgs_width.get_value(),
                            lower=mass.lower,
                            upper=mass.upper,
                        ),
                        "rate": evm.Parameter(
                            model_ggH_Tag0_norm_function(
                                params.mu.get_value(),
                                xs_ggH,
                                br_hgg,
                                eff,
                                lumi,
                            ),
                            name="signal_rate",
                        ),
                        "modifiers": [
                            pm.SymmLogNormalModifier(
                                parameter=params.phoid_syst, kappa=1.05
                            ),
                        ],
                    },
                    "background": {
                        "pdf": pm.Exponential(
                            lambd=params.lamb.get_value(),
                            lower=mass.lower,
                            upper=mass.upper,
                        ),
                        "rate": params.bkg_norm,
                        "modifiers": [],
                    },
                },
                "observations": data,
            }
        }
        return dc

    def dc_to_nll(params, dc):
        nll_terms = []
        for channel_name, channel in dc.items():
            obs_data = channel["observations"]
            pdfs = []
            extended_vals = []
            for proc_name, proc in channel["processes"].items():
                pdf = proc["pdf"]
                rate = proc["rate"]
                for modifier in proc["modifiers"]:
                    rate = modifier.apply(rate)
                pdfs.append(pdf)
                extended_vals.append(rate.get_value())

            sum_pdf = pm.SumPDF(
                pdfs=pdfs,
                extended_vals=extended_vals,
                lower=mass.lower,
                upper=mass.upper,
            )
            nll = pm.create_extended_nll(params, sum_pdf, obs_data)
            nll_terms.append(nll)
        total_nll = jnp.sum(jnp.array(nll_terms))
        return total_nll

    print(dc_to_nll(params, datacard(params)))

    # Unwrap and split for optimization
    params_unwrapped = unwrap(params)
    graphdef, diffable, static = nnx.split(
        params_unwrapped, evm_filter.is_dynamic_parameter, ...
    )

    @nnx.jit
    def loss_fn(dynamic_state, args):
        graphdef, static_state, data = args
        params_unwrapped_local = nnx.merge(
            graphdef, dynamic_state, static_state, copy=True
        )
        _, params_wrapped = wrap_checked(params_unwrapped_local)
        dc = datacard(params_wrapped)
        nll = dc_to_nll(params_wrapped, dc)
        return nll

    solver = optimistix.BFGS(
        rtol=1e-5, atol=1e-7, verbose=frozenset({"step_size", "loss"})
    )
    fitresult = optimistix.minimise(
        loss_fn,
        solver,
        diffable,
        has_aux=False,
        args=(graphdef, static, data),
        options={},
        max_steps=1000,
        throw=True,
    )
    fitted_unwrapped = nnx.merge(graphdef, fitresult.value, static, copy=True)
    fitted_params = wrap(fitted_unwrapped)

    print(f"Final estimate: r = {fitted_params.mu.get_value()}\n")

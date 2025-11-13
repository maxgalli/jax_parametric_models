import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
from scipy import interpolate
from pathlib import Path
import jax.numpy as jnp
import evermore as evm
from flax import nnx
import jax
import optax
from evermore.parameters.transform import MinuitTransform, unwrap, wrap
from evermore.parameters import filter as evm_filter
from jax.experimental import checkify
import optimistix
from jax.flatten_util import ravel_pytree


from paramore.distributions import (
    Exponential,
    Gaussian,
    ExtendedNLL,
    SumPDF,
    ParameterizedFunction,
)
from paramore.modifiers import SymmLogNormalModifier, AsymmetricLogNormalModifier
from paramore.sampling import sample_toys_from_model
from plotting_helpers import plot_as_data, save_image

# Import common classes and functions from api_example
from api_example import (
    Params,
    MeanFunctionWithScale,
    SigmaFunctionWithSmear,
    SignalRate,
    build_model,
)


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

    params = Params(
        evm.Parameter(
            value=125.0, name="higgs_mass", lower=120.0, upper=130.0, frozen=True
        ),
        evm.Parameter(
            value=0.000848571,
            name="d_higgs_mass",
            lower=-5.0,
            upper=5.0,
            transform=minuit_transform,
            frozen=True,
        ),
        evm.Parameter(
            value=1.99705,
            name="higgs_width",
            lower=1.0,
            upper=5.0,
            transform=minuit_transform,
            frozen=True,
        ),
        evm.Parameter(
            value=0.1,
            name="lamb",
            lower=0.0,
            upper=1.0,
            transform=minuit_transform,
        ),
        evm.Parameter(
            value=float(df_data.shape[0]),
            name="bkg_norm",
            lower=0.0,
            upper=1e6,
            transform=minuit_transform,
        ),
        evm.Parameter(
            value=1.0,
            name="mu",
            lower=0.0,
            upper=10.0,
            transform=minuit_transform,
        ),
        evm.NormalParameter(
            value=0.0, name="phoid_syst", transform=minuit_transform
        ),
        evm.NormalParameter(
            value=0.0, name="jec_syst", transform=minuit_transform
        ),
        evm.NormalParameter(
            value=0.0,
            name="nuisance_scale",
            lower=-5.0,
            upper=5.0,
            transform=minuit_transform,
        ),
        evm.NormalParameter(
            value=0.0,
            name="nuisance_smear",
            lower=-5.0,
            upper=5.0,
            transform=minuit_transform,
        )
    )

    params_unwrapped = unwrap(params)
    graphdef, diffable, static = nnx.split(
        params_unwrapped, evm_filter.is_dynamic_parameter, ...
    )

    def loss_fn(dynamic_state, args):
        graphdef, static_state, data, mass, xs_ggH, br_hgg, eff, lumi = args
        params_unwrapped = nnx.merge(
            graphdef, dynamic_state, static_state, copy=True
        )
        _, params_wrapped = wrap_checked(params_unwrapped)
        model = build_model(params_wrapped, mass, xs_ggH, br_hgg, eff, lumi)
        nll = ExtendedNLL(model=model)
        return nll(data)

    def loss_fn_masked(dynamic_state, args):
        """Loss function that handles masked/padded data.

        The mask is used to ignore padded events (NaN values) in the NLL calculation.
        We cannot use boolean indexing (dynamic shapes), so we compute the NLL
        with the mask applied to the sum.
        """
        graphdef, static_state, data, mask, mass, xs_ggH, br_hgg, eff, lumi = args

        # Build model
        params_unwrapped = nnx.merge(
            graphdef, dynamic_state, static_state, copy=True
        )
        _, params_wrapped = wrap_checked(params_unwrapped)
        model = build_model(params_wrapped, mass, xs_ggH, br_hgg, eff, lumi)

        # Compute masked extended NLL
        # N = number of real events (sum of mask)
        N = jnp.sum(mask)
        nu_total = model.extended.value

        # Compute PDF on all data (including padded)
        pdf = model.prob(data)

        # Mask the log probabilities: padded events contribute 0 to the sum
        # Use jnp.where to avoid NaN propagation from padded events
        masked_log_pdf = jnp.where(mask, jnp.log(pdf + 1e-8), 0.0)

        # Extended NLL with Poisson term
        poisson_term = -nu_total + N * jnp.log(nu_total)
        log_likelihood = poisson_term + jnp.sum(masked_log_pdf)

        # Add priors (same as ExtendedNLL)
        # We need to collect parameters with priors and add their log_prob contributions
        prior_total = jnp.array(0.0)
        # Traverse the model to find parameters with priors
        for path, value in nnx.iter_graph(model):
            if isinstance(value, evm.Parameter):
                if getattr(value, "prior", None) is not None:
                    prior_val = value.prior.log_prob(value.value)
                    prior_total = prior_total + jnp.sum(prior_val)

        log_likelihood += prior_total
        return jnp.squeeze(-log_likelihood)

    ntoys = 1000
    model = build_model(params, mass, xs_ggH, br_hgg, eff, lumi)
    # toys is a list of ntoys arrays, each with Poisson-fluctuated event count
    # parameters are sampled from priors for each toy
    print(f"Generating {ntoys} toy datasets...")
    toys = sample_toys_from_model(model, ntoys=ntoys, key=jax.random.PRNGKey(42))

    # ============================================================================
    # STEP 1: Pad toys to uniform length for vmapping
    # ============================================================================
    # Each toy has a different number of events due to Poisson fluctuation.
    # To vmap over toys, we need all toys to have the same shape.
    # Strategy: Pad all toys to max_events length with NaN, create binary masks.

    print(f"Padding {ntoys} toys for parallel fitting...")

    # Find maximum number of events across all toys
    event_counts = jnp.array([len(toy) for toy in toys])
    max_events = int(jnp.max(event_counts))
    print(f"  Event counts: min={int(jnp.min(event_counts))}, max={max_events}, mean={float(jnp.mean(event_counts)):.1f}")

    # Pad each toy to max_events length with NaN (padding value)
    padded_toys = []
    masks = []
    for toy in toys:
        n_events = len(toy)
        # Pad with NaN to reach max_events
        padded = jnp.pad(toy, (0, max_events - n_events), constant_values=jnp.nan)
        # Create binary mask: 1 for real events, 0 for padding
        mask = jnp.concatenate([jnp.ones(n_events), jnp.zeros(max_events - n_events)])
        padded_toys.append(padded)
        masks.append(mask)

    # Stack into arrays of shape (ntoys, max_events)
    padded_toys = jnp.stack(padded_toys)
    masks = jnp.stack(masks)

    # ============================================================================
    # STEP 2: Define single-toy fit function for vmapping
    # ============================================================================

    solver = optimistix.BFGS(
        rtol=1e-5, atol=1e-7, verbose=frozenset()
    )

    def fit_single_toy(toy_data, mask):
        """Fit a single toy dataset.

        This function will be vmapped over all toys for parallel execution.

        Args:
            toy_data: Padded array of shape (max_events,)
            mask: Binary mask of shape (max_events,) indicating valid events

        Returns:
            Fitted parameter values as a pytree
        """
        fitresult = optimistix.minimise(
            loss_fn_masked,
            solver,
            diffable,
            has_aux=False,
            args=(graphdef, static, toy_data, mask, mass, xs_ggH, br_hgg, eff, lumi),
            options={},
            max_steps=1000,
            throw=False,
        )
        return fitresult.value

    # ============================================================================
    # STEP 3: Vmap the fit function over all toys for parallel execution
    # ============================================================================

    print(f"Running {ntoys} toy fits in parallel...")

    # Vmap over the first axis (ntoys dimension)
    # This executes all fits in parallel
    fit_all_toys = jax.vmap(fit_single_toy, in_axes=(0, 0))

    # Execute all fits in parallel
    all_fitted_values = fit_all_toys(padded_toys, masks)

    # ============================================================================
    # STEP 4: Extract fitted parameter values from results
    # ============================================================================

    print("Extracting fitted parameter values...")

    # Convert fitted values back to wrapped parameters for each toy
    fitted_mu_values = []
    fitted_bkg_norm_values = []
    fitted_lamb_values = []
    fitted_phoid_syst_values = []
    fitted_jec_syst_values = []
    fitted_nuisance_scale_values = []
    fitted_nuisance_smear_values = []

    for itoy in range(ntoys):
        # Merge graphdef with fitted values for this toy
        fitted_unwrapped = nnx.merge(
            graphdef, jax.tree.map(lambda x: x[itoy], all_fitted_values), static, copy=True
        )
        fitted_params = wrap(fitted_unwrapped)

        # Extract and store fitted values
        fitted_mu_values.append(float(fitted_params.mu.value))
        fitted_bkg_norm_values.append(float(fitted_params.bkg_norm.value))
        fitted_lamb_values.append(float(fitted_params.lamb.value))
        fitted_phoid_syst_values.append(float(fitted_params.phoid_syst.value))
        fitted_jec_syst_values.append(float(fitted_params.jec_syst.value))
        fitted_nuisance_scale_values.append(float(fitted_params.nuisance_scale.value))
        fitted_nuisance_smear_values.append(float(fitted_params.nuisance_smear.value))

    # Convert to numpy arrays for statistics
    fitted_mu_values = np.array(fitted_mu_values)
    fitted_bkg_norm_values = np.array(fitted_bkg_norm_values)
    fitted_lamb_values = np.array(fitted_lamb_values)
    fitted_phoid_syst_values = np.array(fitted_phoid_syst_values)
    fitted_jec_syst_values = np.array(fitted_jec_syst_values)
    fitted_nuisance_scale_values = np.array(fitted_nuisance_scale_values)
    fitted_nuisance_smear_values = np.array(fitted_nuisance_smear_values)

    # Compute statistics across toys
    print("\n" + "="*60)
    print("Toy fit results (mean ± std across toys):")
    print("="*60)
    print(
        f"r = {fitted_mu_values.mean():.6f} ± {fitted_mu_values.std():.6f}"
    )
    print(
        f"bkg_norm = {fitted_bkg_norm_values.mean():.6f} ± {fitted_bkg_norm_values.std():.6f}"
    )
    print(
        f"lamb = {fitted_lamb_values.mean():.6f} ± {fitted_lamb_values.std():.6f}"
    )
    print(
        f"phoid_syst = {fitted_phoid_syst_values.mean():.6f} ± {fitted_phoid_syst_values.std():.6f}"
    )
    print(
        f"jec_syst = {fitted_jec_syst_values.mean():.6f} ± {fitted_jec_syst_values.std():.6f}"
    )
    print(
        f"nuisance_scale = {fitted_nuisance_scale_values.mean():.6f} ± {fitted_nuisance_scale_values.std():.6f}"
    )
    print(
        f"nuisance_smear = {fitted_nuisance_smear_values.mean():.6f} ± {fitted_nuisance_smear_values.std():.6f}"
    )
    print("="*60)
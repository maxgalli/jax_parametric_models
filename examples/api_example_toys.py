"""Toy study using paramore.

This script demonstrates:
1. Generating toy datasets with parameter sampling from priors
2. Parallel fitting of toys using vmap

Parallelization strategy:
- Prior sampling: vmapped (fixed output shape)
- Toy generation: sequential loop (variable-length arrays per toy)
- Toy fitting: vmapped (after padding toys to uniform length)
"""
import pandas as pd
import numpy as np
import jax.numpy as jnp
import evermore as evm
from flax import nnx
import jax
from evermore.parameters.transform import MinuitTransform, unwrap, wrap
from jax.experimental import checkify
from evermore.parameters import filter as evm_filter
from pathlib import Path
import optimistix
import time

# Import from paramore
import paramore as pm

# Import common classes from api_example
from api_example import (
    Params,
    MeanFunctionWithScale,
    SigmaFunctionWithSmear,
    SignalRate,
)


wrap_checked = checkify.checkify(wrap)

# Enable double precision
jax.config.update("jax_enable_x64", True)


def sample_toys_from_model(
    params,
    mass,
    xs_ggH,
    br_hgg,
    eff,
    lumi,
    ntoys: int,
    key,
):
    """Sample toy datasets from the model.

    PARALLELIZATION NOTE: This function uses a sequential loop over toys
    because each toy has a variable number of events (Poisson-sampled).
    JAX vmap requires fixed shapes, so we cannot vmap the toy generation itself.
    However, prior sampling IS vmapped for efficiency.

    Args:
        params: Params PyTree with parameters
        mass: Observable parameter
        xs_ggH, br_hgg, eff, lumi: Constants
        ntoys: Number of toy datasets
        key: JAX random key

    Returns:
        List of toy datasets (variable-length arrays)
    """
    # Collect parameters with priors
    params_with_priors = []
    for path, value in nnx.iter_graph(params):
        if isinstance(value, evm.Parameter):
            if getattr(value, "prior", None) is not None:
                params_with_priors.append(value)

    # PARALLELIZATION: Vmap prior sampling for all toys at once
    sampled_values = {}
    for param in params_with_priors:
        key, subkey = jax.random.split(key)
        keys = jax.random.split(subkey, ntoys)
        # Vmap: sample from prior for all toys in parallel
        samples = jax.vmap(lambda k: param.prior.sample(key=k, shape=()))(keys)
        sampled_values[param.name] = samples

    # NO PARALLELIZATION: Sequential loop over toys
    toys = []
    keys = jax.random.split(key, ntoys)

    for itoy in range(ntoys):
        # Create a copy of params
        graphdef, state = nnx.split(params)
        params_copy = nnx.merge(graphdef, state, copy=True)

        # Update parameters with sampled values
        for param in params_with_priors:
            sampled_val = sampled_values[param.name][itoy]
            # Update parameter value in place
            for path, value in nnx.iter_graph(params_copy):
                if isinstance(value, evm.Parameter) and value.name == param.name:
                    value.value = sampled_val

        # Build model with updated parameters using ParameterizedFunctions
        signal_mu_func = MeanFunctionWithScale(
            params_copy.higgs_mass,
            params_copy.d_higgs_mass,
            params_copy.nuisance_scale,
        )
        signal_sigma_func = SigmaFunctionWithSmear(
            params_copy.higgs_width,
            params_copy.nuisance_smear,
        )

        signal_pdf = pm.Gaussian(
            mu=signal_mu_func.value,
            sigma=signal_sigma_func.value,
            lower=mass.lower,
            upper=mass.upper,
        )

        background_pdf = pm.Exponential(
            lambd=params_copy.lamb.value,
            lower=mass.lower,
            upper=mass.upper,
        )

        # Compute signal rate with modifiers
        signal_rate_func = SignalRate(params_copy.mu, xs_ggH, br_hgg, eff, lumi)

        phoid_modifier = pm.SymmLogNormalModifier(
            parameter=params_copy.phoid_syst, kappa=1.05
        )
        signal_rate_with_phoid = phoid_modifier.apply(signal_rate_func)

        jec_modifier = pm.AsymmetricLogNormalModifier(
            parameter=params_copy.jec_syst,
            kappa_up=1.056,
            kappa_down=0.951,
        )
        signal_rate_with_all_modifiers = jec_modifier.apply(signal_rate_with_phoid)

        signal_rate = signal_rate_with_all_modifiers.value
        bkg_rate = params_copy.bkg_norm.value

        # Create SumPDF and sample with Poisson fluctuation
        sum_pdf = pm.SumPDF(
            pdfs=[signal_pdf, background_pdf],
            extended_vals=[signal_rate, bkg_rate],
            lower=mass.lower,
            upper=mass.upper,
        )

        # Sample from SumPDF with Poisson fluctuation
        toy_data = sum_pdf.sample_extended(keys[itoy])

        toys.append(toy_data)

    return toys


if __name__ == "__main__":
    minuit_transform = MinuitTransform()

    # Load data and setup parameters
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

    # Observable parameter
    true_mean = 125.0
    mass = evm.Parameter(
        value=true_mean,
        name="CMS_hgg_mass",
        lower=100.0,
        upper=180.0,
        frozen=False,
    )

    # Create parameters
    params = Params(
        mass=mass,
        higgs_mass=evm.Parameter(
            value=125.0, name="higgs_mass", lower=120.0, upper=130.0, frozen=True
        ),
        d_higgs_mass=evm.Parameter(
            value=0.000848571,
            name="d_higgs_mass",
            lower=-5.0,
            upper=5.0,
            transform=minuit_transform,
            frozen=True,
        ),
        higgs_width=evm.Parameter(
            value=1.99705,
            name="higgs_width",
            lower=1.0,
            upper=5.0,
            transform=minuit_transform,
            frozen=True,
        ),
        lamb=evm.Parameter(
            value=0.1, name="lamb", lower=0.0, upper=1.0, transform=minuit_transform
        ),
        bkg_norm=evm.Parameter(
            value=float(df_data.shape[0]),
            name="bkg_norm",
            lower=0.0,
            upper=1e6,
            transform=minuit_transform,
        ),
        mu=evm.Parameter(
            value=1.0, name="mu", lower=0.0, upper=10.0, transform=minuit_transform
        ),
        phoid_syst=evm.NormalParameter(
            value=0.0, name="phoid_syst", transform=minuit_transform
        ),
        jec_syst=evm.NormalParameter(
            value=0.0, name="jec_syst", transform=minuit_transform
        ),
        nuisance_scale=evm.NormalParameter(
            value=0.0,
            name="nuisance_scale",
            lower=-5.0,
            upper=5.0,
            transform=minuit_transform,
        ),
        nuisance_smear=evm.NormalParameter(
            value=0.0,
            name="nuisance_smear",
            lower=-5.0,
            upper=5.0,
            transform=minuit_transform,
        ),
    )

    # ========================================================================
    # Generate toy datasets
    # ========================================================================
    ntoys = 100
    print(f"Generating {ntoys} toy datasets...")

    t0 = time.time()
    toys = sample_toys_from_model(
        params, mass, xs_ggH, br_hgg, eff, lumi, ntoys=ntoys,
        key=jax.random.PRNGKey(42)
    )
    t1 = time.time()
    toy_generation_time = t1 - t0
    print(f"✓ Toy generation took: {toy_generation_time:.3f} seconds ({ntoys/toy_generation_time:.2f} toys/sec)")

    # ========================================================================
    # STEP 1: Pad toys to uniform length for vmapping
    # ========================================================================
    print(f"Padding {ntoys} toys for parallel fitting...")

    event_counts = jnp.array([len(toy) for toy in toys])
    max_events = int(jnp.max(event_counts))
    print(
        f"  Event counts: min={int(jnp.min(event_counts))}, "
        f"max={max_events}, mean={float(jnp.mean(event_counts)):.1f}"
    )

    # Pad each toy to max_events length with NaN
    padded_toys = []
    masks = []
    for toy in toys:
        n_events = len(toy)
        padded = jnp.pad(toy, (0, max_events - n_events), constant_values=jnp.nan)
        mask = jnp.concatenate([jnp.ones(n_events), jnp.zeros(max_events - n_events)])
        padded_toys.append(padded)
        masks.append(mask)

    # Stack into arrays of shape (ntoys, max_events)
    padded_toys = jnp.stack(padded_toys)
    masks = jnp.stack(masks)

    # ========================================================================
    # STEP 2: Define masked loss function for toy fits
    # ========================================================================

    # Unwrap params for optimization
    params_unwrapped = unwrap(params)
    graphdef, diffable, static = nnx.split(
        params_unwrapped, evm_filter.is_dynamic_parameter, ...
    )

    def loss_fn_masked(dynamic_state, args):
        """Masked loss function for toy fits."""
        graphdef, static_state, data, mask, mass, xs_ggH, br_hgg, eff, lumi = args

        # Reconstruct wrapped parameters
        params_unwrapped_local = nnx.merge(
            graphdef, dynamic_state, static_state, copy=True
        )
        _, params_wrapped = wrap_checked(params_unwrapped_local)

        # Build PDFs with current parameters
        signal_mu_func = MeanFunctionWithScale(
            params_wrapped.higgs_mass,
            params_wrapped.d_higgs_mass,
            params_wrapped.nuisance_scale,
        )
        signal_sigma_func = SigmaFunctionWithSmear(
            params_wrapped.higgs_width,
            params_wrapped.nuisance_smear,
        )

        signal_pdf = pm.Gaussian(
            mu=signal_mu_func.value,
            sigma=signal_sigma_func.value,
            lower=mass.lower,
            upper=mass.upper,
        )

        background_pdf = pm.Exponential(
            lambd=params_wrapped.lamb.value,
            lower=mass.lower,
            upper=mass.upper,
        )

        # Compute signal rate with modifiers
        signal_rate_func = SignalRate(params_wrapped.mu, xs_ggH, br_hgg, eff, lumi)
        phoid_modifier = pm.SymmLogNormalModifier(
            parameter=params_wrapped.phoid_syst, kappa=1.05
        )
        signal_rate_with_phoid = phoid_modifier.apply(signal_rate_func)

        jec_modifier = pm.AsymmetricLogNormalModifier(
            parameter=params_wrapped.jec_syst,
            kappa_up=1.056,
            kappa_down=0.951,
        )
        signal_rate_with_all_modifiers = jec_modifier.apply(signal_rate_with_phoid)
        signal_rate = signal_rate_with_all_modifiers.value

        bkg_rate = params_wrapped.bkg_norm.value

        # Create SumPDF
        sum_pdf = pm.SumPDF(
            pdfs=[signal_pdf, background_pdf],
            extended_vals=[signal_rate, bkg_rate],
            lower=mass.lower,
            upper=mass.upper,
        )

        # Compute masked extended NLL
        N = jnp.sum(mask)
        nu_total = signal_rate + bkg_rate

        # Poisson term
        poisson_term = -nu_total + N * jnp.log(nu_total)

        # Get probabilities and mask
        sum_probs = sum_pdf.prob(data)
        masked_log_pdf = jnp.where(mask, jnp.log(sum_probs + 1e-8), 0.0)

        # Log-likelihood
        log_likelihood = poisson_term + jnp.sum(masked_log_pdf)

        # Add priors
        constraints = evm.loss.get_log_probs(params_wrapped)
        prior_values = [v for v in constraints.values()]
        if prior_values:
            prior_total = jnp.sum(jnp.array(prior_values))
            log_likelihood += prior_total

        return jnp.squeeze(-log_likelihood)

    # ========================================================================
    # STEP 3: Define single-toy fit function for vmapping
    # ========================================================================

    solver = optimistix.BFGS(rtol=1e-5, atol=1e-7, verbose=frozenset())

    def fit_single_toy(toy_data, mask):
        """Fit a single toy dataset (will be vmapped)."""
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

    # ========================================================================
    # STEP 4: Vmap the fit function over all toys
    # ========================================================================
    print(f"Running {ntoys} toy fits in parallel...")

    # Vmap over the first axis (ntoys dimension)
    fit_all_toys = jax.vmap(fit_single_toy, in_axes=(0, 0))

    # Execute all fits in parallel
    t0 = time.time()
    all_fitted_values = fit_all_toys(padded_toys, masks)
    # Force computation to finish (JAX is lazy)
    all_fitted_values = jax.tree.map(lambda x: x.block_until_ready(), all_fitted_values)
    t1 = time.time()
    fitting_time = t1 - t0
    print(f"✓ Toy fitting took: {fitting_time:.3f} seconds")

    # ========================================================================
    # STEP 5: Extract fitted parameter values
    # ========================================================================

    print("Extracting fitted parameter values...")

    # Convert fitted values back to wrapped parameters
    fitted_mu_values = []
    fitted_bkg_norm_values = []
    fitted_lamb_values = []
    fitted_phoid_syst_values = []
    fitted_jec_syst_values = []
    fitted_nuisance_scale_values = []
    fitted_nuisance_smear_values = []

    for itoy in range(ntoys):
        fitted_unwrapped = nnx.merge(
            graphdef,
            jax.tree.map(lambda x: x[itoy], all_fitted_values),
            static,
            copy=True,
        )
        fitted_params = wrap(fitted_unwrapped)

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

    # ========================================================================
    # Print results
    # ========================================================================

    print("\n" + "=" * 60)
    print("Toy fit results (mean ± std across toys):")
    print("=" * 60)
    print(f"r = {fitted_mu_values.mean():.6f} ± {fitted_mu_values.std():.6f}")
    print(
        f"bkg_norm = {fitted_bkg_norm_values.mean():.6f} ± {fitted_bkg_norm_values.std():.6f}"
    )
    print(f"lamb = {fitted_lamb_values.mean():.6f} ± {fitted_lamb_values.std():.6f}")
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
    print("=" * 60)

    print("\n" + "=" * 60)
    print("TIMING SUMMARY:")
    print("=" * 60)
    print(f"Toy generation: {toy_generation_time:.3f} seconds")
    print(f"Toy fitting:    {fitting_time:.3f} seconds")
    print(f"Total time:     {toy_generation_time + fitting_time:.3f} seconds")
    print("=" * 60)

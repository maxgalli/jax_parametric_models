"""Toy study using paramore with vmapped toy generation.

This script demonstrates:
1. Generating toy datasets with vmapped parameter sampling and event generation
2. Parallel fitting of toys using vmap
3. Working with PyTrees directly in vmapped functions

Parallelization strategy:
- Prior sampling: vmapped evm.sample.sample_from_priors (one call per toy)
- Toy generation: vmapped sample_single_toy (works with PyTree parameters)
- Toy fitting: vmapped fit_single_toy (uniform array length via masks)

The implementation uses JAX's native PyTree handling - parameters are passed
as PyTrees through vmap, and JAX automatically vectorizes over the leading
dimension of arrays within the PyTree structure.
"""

import time
from pathlib import Path

import evermore as evm
import jax
import jax.numpy as jnp
import numpy as np
import optimistix
import pandas as pd

# Import common classes from combine_gammagamma
from combine_gammagamma import Params
from evermore.parameters import filter as evm_filter
from evermore.parameters.transform import MinuitTransform, unwrap, wrap
from flax import nnx
from jax.experimental import checkify

# Import from paramore
import paramore as pm

wrap_checked = checkify.checkify(wrap)

# Enable double precision
jax.config.update("jax_enable_x64", True)


def sample_single_toy(key, params_pytree, mass, xs_ggH, br_hgg, eff, lumi, max_events):
    """Sample a single toy dataset (to be vmapped).

    Args:
        key: JAX random key
        params_pytree: PyTree of parameter values (from evm.sample.sample_from_priors)
        mass, xs_ggH, br_hgg, eff, lumi: Constants
        max_events: Maximum number of events to sample

    Returns:
        samples: (max_events,) array
        mask: (max_events,) boolean array
    """
    # Build PDFs with sampled parameters
    signal_mu = (
        params_pytree.higgs_mass.get_value() + params_pytree.d_higgs_mass.get_value()
    ) * (1.0 + 0.003 * params_pytree.nuisance_scale.get_value())
    signal_sigma = params_pytree.higgs_width.get_value() * (
        1.0 + 0.045 * params_pytree.nuisance_smear.get_value()
    )

    signal_pdf = pm.Gaussian(
        mu=signal_mu,
        sigma=signal_sigma,
        lower=mass.lower,
        upper=mass.upper,
    )

    background_pdf = pm.Exponential(
        lambd=params_pytree.lamb.get_value(),
        lower=mass.lower,
        upper=mass.upper,
    )

    # Compute signal rate with modifiers
    signal_rate_base = evm.Parameter(
        value=params_pytree.mu.get_value() * xs_ggH * br_hgg * eff * lumi,
        name="signal_rate_base",
    )

    # Apply modifiers
    phoid_modifier = pm.SymmLogNormalModifier(
        parameter=params_pytree.phoid_syst, kappa=1.05
    )

    jec_modifier = pm.AsymmetricLogNormalModifier(
        parameter=params_pytree.jec_syst,
        kappa_up=1.056,
        kappa_down=0.951,
    )
    composed_modifier = pm.ComposedModifier(phoid_modifier, jec_modifier)
    signal_rate_with_all_modifiers = composed_modifier.apply(signal_rate_base)

    signal_rate = signal_rate_with_all_modifiers.get_value()
    bkg_rate = params_pytree.bkg_norm.get_value()

    # Create SumPDF and sample with fixed size
    sum_pdf = pm.SumPDF(
        pdfs=[signal_pdf, background_pdf],
        extended_vals=[signal_rate, bkg_rate],
        lower=mass.lower,
        upper=mass.upper,
    )

    # Sample with fixed output size
    samples, mask = sum_pdf.sample_extended_fixed(key, max_events)

    return samples, mask


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
    # Calculate max_events for toy generation
    # ========================================================================

    # Compute nominal expected events
    nominal_signal_rate = params.mu.get_value() * xs_ggH * br_hgg * eff * lumi
    nominal_bkg_rate = params.bkg_norm.get_value()
    expected_total = nominal_signal_rate + nominal_bkg_rate

    # Add 6 sigma buffer
    max_events = int(expected_total + 6 * jnp.sqrt(expected_total))
    print(f"Expected total events: {expected_total:.1f}")
    print(f"Max events (expected + 6σ): {max_events}")

    # ========================================================================
    # Generate toy datasets using vmap
    # ========================================================================
    ntoys = 1000
    print(f"\nGenerating {ntoys} toy datasets (vmapped)...")

    t0 = time.time()

    # Sample from priors using evermore's built-in function
    key = jax.random.PRNGKey(42)
    key_prior, key_toys = jax.random.split(key)

    # Split keys for prior sampling (one per toy)
    prior_keys = jax.random.split(key_prior, ntoys)
    # Split keys for toy generation (one per toy)
    toy_keys = jax.random.split(key_toys, ntoys)

    # Vmap over sample_from_priors to get a batch of parameter samples
    # Each call produces a PyTree with scalar .get_value() for each parameter
    # Vmapping creates a PyTree where each .get_value() has shape (ntoys,)
    def sample_params_for_toy(rng_key):
        """Sample parameters for one toy using nnx.Rngs."""
        rngs = nnx.Rngs(default=rng_key)
        return evm.sample.sample_from_priors(rngs, params)

    sampled_params_batched = jax.vmap(sample_params_for_toy)(prior_keys)

    # Now vmap over toy generation
    # sampled_params_batched is a PyTree where each parameter's .get_value() has shape (ntoys,)
    # We need to index into it for each toy
    def sample_toy_with_params(key, param_tree_batch, idx):
        """Extract single toy params and generate toy dataset."""
        # Extract parameters for this toy
        single_toy_params = jax.tree.map(lambda p: p[idx], param_tree_batch)
        return sample_single_toy(
            key, single_toy_params, mass, xs_ggH, br_hgg, eff, lumi, max_events
        )

    toys, masks = jax.vmap(
        lambda k, idx: sample_toy_with_params(k, sampled_params_batched, idx),
        in_axes=(0, 0),
    )(toy_keys, jnp.arange(ntoys))

    t1 = time.time()
    toy_generation_time = t1 - t0
    print(
        f"✓ Toy generation took: {toy_generation_time:.3f} seconds ({ntoys / toy_generation_time:.2f} toys/sec)"
    )

    # ========================================================================
    # Summary of toy event counts
    # ========================================================================
    event_counts = jnp.sum(masks, axis=1)
    print("\nToy event counts:")
    print(
        f"  min={int(jnp.min(event_counts))}, "
        f"max={int(jnp.max(event_counts))}, "
        f"mean={float(jnp.mean(event_counts)):.1f}"
    )

    # ========================================================================
    # STEP 2: Define masked loss function for toy fits
    # ========================================================================

    # Unwrap params for optimization
    params_unwrapped = unwrap(params)
    graphdef, diffable, static = nnx.split(
        params_unwrapped, evm_filter.is_dynamic_parameter, ...
    )

    @nnx.jit
    def loss_fn_masked(dynamic_state, args):
        """Masked loss function for toy fits."""
        graphdef, static_state, data, mask, mass, xs_ggH, br_hgg, eff, lumi = args

        # Reconstruct wrapped parameters
        params_unwrapped_local = nnx.merge(
            graphdef, dynamic_state, static_state, copy=True
        )
        _, params_wrapped = wrap_checked(params_unwrapped_local)

        # Build PDFs with current parameters
        signal_mu = (
            params_wrapped.higgs_mass.get_value()
            + params_wrapped.d_higgs_mass.get_value()
        ) * (1.0 + 0.003 * params_wrapped.nuisance_scale.get_value())
        signal_sigma = params_wrapped.higgs_width.get_value() * (
            1.0 + 0.045 * params_wrapped.nuisance_smear.get_value()
        )

        signal_pdf = pm.Gaussian(
            mu=signal_mu,
            sigma=signal_sigma,
            lower=mass.lower,
            upper=mass.upper,
        )

        background_pdf = pm.Exponential(
            lambd=params_wrapped.lamb.get_value(),
            lower=mass.lower,
            upper=mass.upper,
        )

        # Compute signal rate with modifiers
        signal_rate_base = evm.Parameter(
            value=params_wrapped.mu.get_value() * xs_ggH * br_hgg * eff * lumi,
            name="signal_rate_base",
        )
        phoid_modifier = pm.SymmLogNormalModifier(
            parameter=params_wrapped.phoid_syst, kappa=1.05
        )

        jec_modifier = pm.AsymmetricLogNormalModifier(
            parameter=params_wrapped.jec_syst,
            kappa_up=1.056,
            kappa_down=0.951,
        )
        composed_modifier = pm.ComposedModifier(phoid_modifier, jec_modifier)
        signal_rate_with_all_modifiers = composed_modifier.apply(signal_rate_base)
        signal_rate = signal_rate_with_all_modifiers.get_value()

        bkg_rate = params_wrapped.bkg_norm.get_value()

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

    @nnx.jit
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
    print(f"\nRunning {ntoys} toy fits in parallel...")

    # Vmap over the first axis (ntoys dimension)
    fit_all_toys = jax.vmap(fit_single_toy, in_axes=(0, 0))

    # Execute all fits in parallel
    t0 = time.time()
    all_fitted_values = fit_all_toys(toys, masks)
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

        fitted_mu_values.append(float(fitted_params.mu.get_value()))
        fitted_bkg_norm_values.append(float(fitted_params.bkg_norm.get_value()))
        fitted_lamb_values.append(float(fitted_params.lamb.get_value()))
        fitted_phoid_syst_values.append(float(fitted_params.phoid_syst.get_value()))
        fitted_jec_syst_values.append(float(fitted_params.jec_syst.get_value()))
        fitted_nuisance_scale_values.append(
            float(fitted_params.nuisance_scale.get_value())
        )
        fitted_nuisance_smear_values.append(
            float(fitted_params.nuisance_smear.get_value())
        )

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

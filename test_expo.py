import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import os
import pandas as pd

# === Exponential Model ===
class ExponentialModel(eqx.Module):
    """Exponential distribution model with a learnable rate parameter lambda."""
    lambda_: jax.Array  # Directly optimize lambda

    def __init__(self, init_lambda: float):
        self.lambda_ = jnp.array(init_lambda)  # Direct assignment

    def pdf(self, x: jax.Array) -> jax.Array:
        """Probability density function of an exponential distribution."""
        #return self.lambda_ * jnp.exp(-self.lambda_ * x)
        #return jnp.exp(-self.lambda_ * x)
        constant = 1000.0
        factor = (jnp.exp(constant * jnp.max(x)) - jnp.exp(constant * jnp.min(x))) / constant
        #return factor * jnp.exp(self.lambda_ * x)
        return jnp.exp(self.lambda_ * x)

# === Negative Log-Likelihood ===
def negative_log_likelihood(model: ExponentialModel, data: jax.Array) -> jax.Array:
    """Compute the negative log-likelihood for the exponential distribution."""
    #lambda_ = jnp.clip(model.lambda_, 1e-6, 10.0)  # Ensure lambda stays positive
    lambda_ = model.lambda_
    #return -jnp.sum(jnp.log(lambda_) - lambda_ * data)
    return -jnp.sum(-lambda_*data)

# === Synthetic Data Generation ===
key = jax.random.PRNGKey(0)
true_lambda = 2.0  # True rate parameter
num_samples = 100
#data = jax.random.exponential(key, shape=(num_samples,)) / true_lambda  # Exponential-distributed data
data_dir = "../StatsStudies/ExercisesForCourse/Hgg_zfit/data"
fl_data = os.path.join(data_dir, "data_part1.parquet")
df_data = pd.read_parquet(fl_data)
var_name = "CMS_hgg_mass"
#df_data_sides = df_data[(df_data[var_name] > 100) & (df_data[var_name] < 115) | (df_data[var_name] > 135) & (df_data[var_name] < 180)]
df_data = df_data[(df_data[var_name] > 100) & (df_data[var_name] < 180)]
#data = jax.numpy.array(df_data_sides[var_name].values)
data = jax.numpy.array(df_data[var_name].values)

# === Model Initialization ===
init_lambda = 0.05  # Initial guess
model = ExponentialModel(init_lambda)

# === Optimizer (Adam) ===
learning_rate = 0.1
optimizer = optax.adam(learning_rate)
#optimizer = optax.adamw(learning_rate)
opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))  # Optimize lambda_

# === Training Step ===
@eqx.filter_jit
def step(model, opt_state, data):
    print(model.lambda_)
    loss, grads = jax.value_and_grad(negative_log_likelihood)(model, data)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)

    # Clip lambda_ to keep it positive
    model = eqx.tree_at(lambda m: m.lambda_, model, jnp.clip(model.lambda_, 1e-6, 10.0))

    return model, opt_state, loss

# === Training Loop ===
num_epochs = 10000
for epoch in range(num_epochs):
    model, opt_state, loss = step(model, opt_state, data)
    if epoch % 50 == 0:
        print(f"Epoch {epoch}: Loss = {loss:.4f}, Lambda = {model.lambda_:.4f}")

# === Final Estimated Parameter ===
print(f"Final estimated lambda: {model.lambda_:.4f} (True value: {true_lambda})")

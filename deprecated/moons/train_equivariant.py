import numpy as np
import jax
import jax.numpy as jnp
import e3nn_jax as e3nn
from models import EquivariantMLP
import optax
from e3nn_jax import IrrepsArray

def mse_loss(params, model, batch_input, batch_target):
    mu, sigma_sq = model.apply(params, batch_input)
    mu_array = mu.array
    target_array = batch_target.array
    return jnp.mean((mu_array - target_array) ** 2)

def nll_loss(params, model, batch_input, batch_target):
    mu, sigma_sq = model.apply(params, batch_input)
    mu_array = mu.array
    sigma_sq_array = sigma_sq.array
    target_array = batch_target.array
    return jnp.mean(((mu_array - target_array) ** 2 / (2 * sigma_sq_array)) + 0.5 * jnp.log(sigma_sq_array))

def beta_nll_loss(params, model, batch_input, batch_target, beta=1.0):
    mu, sigma_sq = model.apply(params, batch_input)
    mu_array = mu.array
    sigma_sq_array = sigma_sq.array
    target_array = batch_target.array
    sigma_sq_array_beta = jax.lax.stop_gradient(sigma_sq_array) ** beta
    return jnp.mean(sigma_sq_array_beta *(((mu_array - target_array) ** 2 / (2 * sigma_sq_array)) + 0.5 * jnp.log(sigma_sq_array)))

def combined_loss(params, model, batch_input, batch_target, beta=1.0):
    return mse_loss(params, model, batch_input, batch_target) + beta_nll_loss(params, model, batch_input, batch_target, beta=beta)

def train_model(model, key, input_locations, output_locations, n_holdout, minimum_epochs, maximum_epochs_no_improve, maximum_epochs, batch_size, train_val_split):
    key, subkey = jax.random.split(key)
    n_samples = input_locations.shape[0]

    n_test = n_holdout
    n_train_val = n_samples - n_test

    n_train = int(n_train_val * train_val_split)
    n_val = n_train_val - n_train

    perm = jax.random.permutation(subkey, jnp.arange(n_samples))
    train_indices = perm[:n_train]
    val_indices = perm[n_train:n_train + n_val]
    test_indices = perm[n_train_val:]

    train_input_locations = e3nn.IrrepsArray(model.input_irreps, input_locations[train_indices])
    train_output_locations = output_locations[train_indices]

    val_input_locations = e3nn.IrrepsArray(model.input_irreps, input_locations[val_indices])
    val_output_locations = output_locations[val_indices]

    test_input_locations = e3nn.IrrepsArray(model.input_irreps, input_locations[test_indices])
    test_output_locations = output_locations[test_indices]

    dummy_input = train_input_locations[:1]
    parameters = model.init(key, dummy_input)
    optimizer = optax.adam(1e-4)
    opt_state = optimizer.init(parameters)

    best_val_loss = jnp.inf
    epochs_no_improve = 0
    
    train_losses = []
    val_losses = []

    for epoch in range(maximum_epochs):
        key, subkey = jax.random.split(key)
        train_loss = 0
        for i in range(0, n_train, batch_size):
            batch_input_locations = train_input_locations[i:i+batch_size]
            batch_output_locations = train_output_locations[i:i+batch_size]
            batch_output_locations = e3nn.IrrepsArray(model.output_irreps, batch_output_locations)

            loss, grad = jax.value_and_grad(combined_loss)(parameters, model, batch_input_locations, batch_output_locations)

            updates, opt_state = optimizer.update(grad, opt_state)
            parameters = optax.apply_updates(parameters, updates)
            train_loss += loss

        train_loss /= n_train

        val_prediction_mu, val_prediction_sigma_sq = model.apply(parameters, val_input_locations)
        val_prediction_mu = val_prediction_mu.array
        val_prediction_sigma_sq = val_prediction_sigma_sq.array
        val_target = val_output_locations
        val_loss = jnp.mean((val_prediction_sigma_sq *((val_prediction_mu - val_target) ** 2 / (2 * val_prediction_sigma_sq)) + 0.5 * jnp.log(val_prediction_sigma_sq))) + jnp.mean((val_prediction_mu - val_target) ** 2)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"Epochs no improvement: {epochs_no_improve}")

        if epochs_no_improve > maximum_epochs_no_improve:
            print(f"Early stopping at epoch {epoch}")
            train_losses = np.array(train_losses)
            val_losses = np.array(val_losses)

            test_prediction_mu, test_prediction_sigma_sq = model.apply(parameters, test_input_locations)
            test_prediction_mu = test_prediction_mu.array
            test_prediction_sigma_sq = test_prediction_sigma_sq.array
            test_target = test_output_locations
            test_loss = jnp.mean(((test_prediction_mu - test_target) ** 2 / (2 * test_prediction_sigma_sq)) + 0.5 * jnp.log(test_prediction_sigma_sq)) + jnp.mean((test_prediction_mu - test_target) ** 2)
            print(f"Test Loss: {test_loss}")

            test_prediction_mu_npy = np.array(test_prediction_mu)
            test_prediction_sigma_sq_npy = np.array(test_prediction_sigma_sq)
            test_target_npy = np.array(test_target)
            test_inputs_npy = np.array(test_input_locations.array)
            np.save("../data/test_prediction_mu_equivariant.npy", test_prediction_mu_npy)
            np.save("../data/test_prediction_sigma_sq_equivariant.npy", test_prediction_sigma_sq_npy)
            np.save("../data/test_target_equivariant.npy", test_target_npy)
            np.save("../data/test_inputs_equivariant.npy", test_inputs_npy)

            np.save("../data/train_losses_equivariant.npy", train_losses)
            np.save("../data/val_losses_equivariant.npy", val_losses)
            val_prediction_mu_npy = np.array(val_prediction_mu)
            val_prediction_sigma_sq_npy = np.array(val_prediction_sigma_sq)
            val_target_npy = np.array(val_target)
            val_inputs_npy = np.array(val_input_locations.array)
            np.save("../data/val_prediction_mu_equivariant.npy", val_prediction_mu_npy)
            np.save("../data/val_prediction_sigma_sq_equivariant.npy", val_prediction_sigma_sq_npy)
            np.save("../data/val_target_equivariant.npy", val_target_npy)
            np.save("../data/val_inputs_equivariant.npy", val_inputs_npy)
            break

        if epoch == maximum_epochs - 1:
            train_losses = np.array(train_losses)
            val_losses = np.array(val_losses)

            test_prediction_mu, test_prediction_sigma_sq = model.apply(parameters, test_input_locations)
            test_prediction_mu = test_prediction_mu.array
            test_prediction_sigma_sq = test_prediction_sigma_sq.array
            test_target = test_output_locations
            test_loss = jnp.mean(((test_prediction_mu - test_target) ** 2 / (2 * test_prediction_sigma_sq)) + 0.5 * jnp.log(test_prediction_sigma_sq))
            print(f"Test Loss: {test_loss}")

            test_prediction_mu_npy = np.array(test_prediction_mu)
            test_prediction_sigma_sq_npy = np.array(test_prediction_sigma_sq)
            test_target_npy = np.array(test_target)
            test_inputs_npy = np.array(test_input_locations.array)
            np.save("../data/test_prediction_mu_equivariant.npy", test_prediction_mu_npy)
            np.save("../data/test_prediction_sigma_sq_equivariant.npy", test_prediction_sigma_sq_npy)
            np.save("../data/test_target_equivariant.npy", test_target_npy)
            np.save("../data/test_inputs_equivariant.npy", test_inputs_npy)

            np.save("../data/train_losses_equivariant.npy", train_losses)
            np.save("../data/val_losses_equivariant.npy", val_losses)
            val_prediction_mu_npy = np.array(val_prediction_mu)
            val_prediction_sigma_sq_npy = np.array(val_prediction_sigma_sq)
            val_target_npy = np.array(val_target)
            val_inputs_npy = np.array(val_input_locations.array)
            np.save("../data/val_prediction_mu_equivariant.npy", val_prediction_mu_npy)
            np.save("../data/val_prediction_sigma_sq_equivariant.npy", val_prediction_sigma_sq_npy)
            np.save("../data/val_target_equivariant.npy", val_target_npy)
            np.save("../data/val_inputs_equivariant.npy", val_inputs_npy)
            print(f"Maximum epochs reached at epoch {epoch}")

        print(f"Epoch: {epoch}, Train Loss: {train_loss}, Val Loss: {val_loss}")



if __name__ == "__main__":
    input_locations = np.load("../data/input_locations_90_degree_offset_100000_samples_0.1_noise.npy")
    output_locations = np.load("../data/output_locations_90_degree_offset_100000_samples_0.1_noise.npy")

    if input_locations.shape[-1] == 2:
        input_locations = np.pad(input_locations, ((0, 0), (0, 1)), mode="constant")
    if output_locations.shape[-1] == 2:
        output_locations = np.pad(output_locations, ((0, 0), (0, 1)), mode="constant")
    
    input_irreps = e3nn.Irreps("1x1e")
    output_irreps = e3nn.Irreps("1x1e")

    model = EquivariantMLP(
            input_irreps=input_irreps, 
            output_irreps=output_irreps, 
            hidden_dim=32,
            )

    key = jax.random.PRNGKey(0)

    n_holdout = 200
    minimum_epochs = 10
    maximum_epochs_no_improve = 5
    maximum_epochs = 50
    batch_size = 128

    input_locations = jnp.array(input_locations)
    output_locations = jnp.array(output_locations)

    train_val_split = 0.8

    train_model(model, key, input_locations, output_locations, n_holdout, minimum_epochs, maximum_epochs_no_improve, maximum_epochs, batch_size, train_val_split)


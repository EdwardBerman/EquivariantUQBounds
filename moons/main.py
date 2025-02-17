import numpy as np
import jax
import jax.numpy as jnp
import e3nn_jax as e3nn
from models import EquivariantMLP
import optax
from e3nn_jax import IrrepsArray

def mse_loss(params, model, batch_input, batch_target):
    prediction = model.apply(params, batch_input)
    pred_array = prediction.array
    target_array = batch_target.array
    return jnp.mean((pred_array - target_array) ** 2)

def train_model(model, key, input_locations, output_locations, n_holdout, minimum_epochs, maximum_epochs_no_improve, maximum_epochs, batch_size, train_val_split):

    key, subkey = jax.random.split(key)
    n_samples = input_locations.shape[0]
    n_train = n_samples - n_holdout
    n_train = int(n_train * train_val_split)
    n_val = n_samples - n_train
    
    train_indices = jax.random.permutation(subkey, jnp.arange(n_samples))[:n_train]
    val_indices = jax.random.permutation(subkey, jnp.arange(n_samples))[n_train:]

    train_input_locations = e3nn.IrrepsArray(model.input_irreps, input_locations[train_indices])
    train_output_locations = output_locations[train_indices]

    val_input_locations = e3nn.IrrepsArray(model.input_irreps, input_locations[val_indices])
    val_output_locations = output_locations[val_indices]

    dummy_input = train_input_locations[:1]
    parameters = model.init(key, dummy_input)
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(parameters)

    best_val_loss = jnp.inf
    epochs_no_improve = 0

    for epoch in range(maximum_epochs):
        key, subkey = jax.random.split(key)
        train_loss = 0
        for i in range(0, n_train, batch_size):
            batch_input_locations = train_input_locations[i:i+batch_size]
            batch_output_locations = train_output_locations[i:i+batch_size]
            batch_output_locations = e3nn.IrrepsArray(model.output_irreps, batch_output_locations)

            loss, grad = jax.value_and_grad(mse_loss)(parameters, model, batch_input_locations, batch_output_locations)

            updates, opt_state = optimizer.update(grad, opt_state)
            parameters = optax.apply_updates(parameters, updates)
            train_loss += loss

        train_loss /= n_train


        val_prediction = model.apply(parameters, val_input_locations).array
        val_target = val_output_locations
        val_loss = jnp.mean((val_prediction - val_target) ** 2)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve > maximum_epochs_no_improve:
            break

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
            hidden_dim=2
            )

    key = jax.random.PRNGKey(0)

    n_holdout = 200
    minimum_epochs = 1000
    maximum_epochs_no_improve = 1000
    maximum_epochs = 10000
    batch_size = 128

    input_locations = jnp.array(input_locations)
    output_locations = jnp.array(output_locations)

    train_val_split = 0.8

    train_model(model, key, input_locations, output_locations, n_holdout, minimum_epochs, maximum_epochs_no_improve, maximum_epochs, batch_size, train_val_split)


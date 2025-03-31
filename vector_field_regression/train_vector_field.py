import numpy as np
import jax
import jax.numpy as jnp
import e3nn_jax as e3nn
from e3nn_jax import IrrepsArray
import optax
import os
import matplotlib.pyplot as plt
from dataclasses import dataclass
from simple_parsing import ArgumentParser
from models import CorrectEquivariantVectorFieldModel, IncorrectEquivariantVectorFieldModel, MLPVectorFieldModel
from matplotlib import rcParams, rc
from matplotlib.gridspec import GridSpec
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes



def set_rc_params(fontsize=None):
    '''
    Set figure parameters
    '''

    if fontsize is None:
        fontsize=16
    else:
        fontsize=int(fontsize)

    rc('font',**{'family':'serif'})
    rc('text', usetex=True)

    #plt.rcParams.update({'figure.facecolor':'w'})
    plt.rcParams.update({'axes.linewidth': 1.3})
    plt.rcParams.update({'xtick.labelsize': fontsize})
    plt.rcParams.update({'ytick.labelsize': fontsize})
    plt.rcParams.update({'xtick.major.size': 8})
    plt.rcParams.update({'xtick.major.width': 1.3})
    plt.rcParams.update({'xtick.minor.visible': True})
    plt.rcParams.update({'xtick.minor.width': 1.})
    plt.rcParams.update({'xtick.minor.size': 6})
    plt.rcParams.update({'xtick.direction': 'out'})
    plt.rcParams.update({'ytick.major.width': 1.3})
    plt.rcParams.update({'ytick.major.size': 8})
    plt.rcParams.update({'ytick.minor.visible': True})
    plt.rcParams.update({'ytick.minor.width': 1.})
    plt.rcParams.update({'ytick.minor.size':6})
    plt.rcParams.update({'ytick.direction':'out'})
    plt.rcParams.update({'axes.labelsize': fontsize})
    plt.rcParams.update({'axes.titlesize': fontsize})
    plt.rcParams.update({'legend.fontsize': int(fontsize-2)})
    plt.rcParams['text.usetex'] = True
    plt.rcParams['text.latex.preamble'] = r'\usepackage{amssymb}'

    return


# Loss functions
def mse_loss(params, model, batch_input, batch_target):
    mu, sigma_sq = model.apply(params, batch_input)
    if isinstance(mu, IrrepsArray):
        mu_array = mu.array
        target_array = batch_target.array if hasattr(batch_target, 'array') else batch_target
    else:
        mu_array = mu
        target_array = batch_target
    return jnp.mean((mu_array - target_array) ** 2)

def nll_loss(params, model, batch_input, batch_target):
    mu, sigma_sq = model.apply(params, batch_input)
    if isinstance(mu, IrrepsArray):
        mu_array = mu.array
        sigma_sq_array = sigma_sq.array
        target_array = batch_target.array if hasattr(batch_target, 'array') else batch_target
    else:
        mu_array = mu
        sigma_sq_array = sigma_sq
        target_array = batch_target
    return jnp.mean(((mu_array - target_array) ** 2 / (2 * sigma_sq_array)) + 0.5 * jnp.log(sigma_sq_array))

def beta_nll_loss(params, model, batch_input, batch_target, beta=1.0):
    mu, sigma_sq = model.apply(params, batch_input)
    if isinstance(mu, IrrepsArray):
        mu_array = mu.array
        sigma_sq_array = sigma_sq.array
        target_array = batch_target.array if hasattr(batch_target, 'array') else batch_target
    else:
        mu_array = mu
        sigma_sq_array = sigma_sq
        target_array = batch_target
    sigma_sq_array_beta = jax.lax.stop_gradient(sigma_sq_array) ** beta
    return jnp.mean(sigma_sq_array_beta *(((mu_array - target_array) ** 2 / (2 * sigma_sq_array)) + 0.5 * jnp.log(sigma_sq_array)))

def combined_loss(params, model, batch_input, batch_target, beta=1.0):
    return mse_loss(params, model, batch_input, batch_target) + beta_nll_loss(params, model, batch_input, batch_target, beta=beta)

def train_model(model, model_type, key, input_positions, vector_field, n_holdout, 
                minimum_epochs, maximum_epochs_no_improve, maximum_epochs, 
                batch_size, train_val_split, beta, save_dir):
    """
    Train a model for vector field regression.
    
    Args:
        model: The model to train
        model_type: Type of model ('correct_equivariant', 'incorrect_equivariant', 'mlp')
        key: JAX random key
        input_positions: Input positions (x,y coordinates)
        vector_field: Vector field values at each position
        n_holdout: Number of samples to hold out for testing
        minimum_epochs: Minimum number of epochs to train
        maximum_epochs_no_improve: Maximum number of epochs with no improvement before early stopping
        maximum_epochs: Maximum number of epochs to train
        batch_size: Batch size for training
        train_val_split: Proportion of non-test data to use for training
        beta: Weight for the beta-NLL loss
        save_dir: Directory to save results
    """
    key, subkey = jax.random.split(key)
    n_samples = input_positions.shape[0]
    
    # Split data into train, validation, and test sets
    n_test = n_holdout
    n_train_val = n_samples - n_test
    
    n_train = int(n_train_val * train_val_split)
    n_val = n_train_val - n_train
    
    perm = jax.random.permutation(subkey, jnp.arange(n_samples))
    train_indices = perm[:n_train]
    val_indices = perm[n_train:n_train + n_val]
    test_indices = perm[n_train_val:]
    
    # Process inputs and targets based on model type
    is_equivariant = model_type in ['correct_equivariant', 'incorrect_equivariant']
    
    if is_equivariant:
        # For equivariant models, we need to pad with zeros for the z-dimension
        # and convert to IrrepsArray
        input_positions_padded = jnp.pad(input_positions, ((0, 0), (0, 1)), mode="constant")
        vector_field_padded = jnp.pad(vector_field, ((0, 0), (0, 1)), mode="constant")
        
        # Set up input and output irreps for 3D (with z=0)
        input_irreps = e3nn.Irreps("1x1e")
        output_irreps = e3nn.Irreps("1x1e")
        
        # Convert to IrrepsArray
        train_inputs = IrrepsArray(input_irreps, input_positions_padded[train_indices])
        train_targets = IrrepsArray(output_irreps, vector_field_padded[train_indices])
        
        val_inputs = IrrepsArray(input_irreps, input_positions_padded[val_indices])
        val_targets = IrrepsArray(output_irreps, vector_field_padded[val_indices])
        
        test_inputs = IrrepsArray(input_irreps, input_positions_padded[test_indices])
        test_targets = IrrepsArray(output_irreps, vector_field_padded[test_indices])
    else:
        # For standard MLP, we use the raw inputs
        train_inputs = input_positions[train_indices]
        train_targets = vector_field[train_indices]
        
        val_inputs = input_positions[val_indices]
        val_targets = vector_field[val_indices]
        
        test_inputs = input_positions[test_indices]
        test_targets = vector_field[test_indices]
    
    # Initialize model and optimizer
    dummy_input = train_inputs[:1]
    parameters = model.init(key, dummy_input)
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(parameters)
    
    # Training loop
    best_val_loss = jnp.inf
    epochs_no_improve = 0
    best_parameters = parameters
    
    train_losses = []
    val_losses = []
    
    for epoch in range(maximum_epochs):
        key, subkey = jax.random.split(key)
        train_loss = 0
        
        # Shuffle training data
        train_indices_epoch = jax.random.permutation(subkey, jnp.arange(n_train))
        
        # Mini-batch training
        for i in range(0, n_train, batch_size):
            batch_indices = train_indices_epoch[i:i+batch_size]
            batch_inputs = train_inputs[batch_indices]
            batch_targets = train_targets[batch_indices]
            
            loss, grad = jax.value_and_grad(combined_loss)(
                parameters, model, batch_inputs, batch_targets, beta)
            
            updates, opt_state = optimizer.update(grad, opt_state)
            parameters = optax.apply_updates(parameters, updates)
            train_loss += loss
        
        train_loss /= np.ceil(n_train / batch_size)
        
        # Validate
        val_prediction_mu, val_prediction_sigma_sq = model.apply(parameters, val_inputs)
        
        # Extract arrays from IrrepsArray if needed
        if isinstance(val_prediction_mu, IrrepsArray):
            val_prediction_mu_array = val_prediction_mu.array
            val_prediction_sigma_sq_array = val_prediction_sigma_sq.array
            val_targets_array = val_targets.array if hasattr(val_targets, 'array') else val_targets
        else:
            val_prediction_mu_array = val_prediction_mu
            val_prediction_sigma_sq_array = val_prediction_sigma_sq
            val_targets_array = val_targets
        
        val_loss = jnp.mean((val_prediction_mu_array - val_targets_array) ** 2)
        
        train_losses.append(float(train_loss))
        val_losses.append(float(val_loss))
        
        # Early stopping check
        if epoch >= minimum_epochs:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                # Save best model parameters
                best_parameters = parameters
            else:
                epochs_no_improve += 1
                
            if epochs_no_improve > maximum_epochs_no_improve:
                print(f"Early stopping at epoch {epoch}")
                parameters = best_parameters  # Use the best parameters
                break
        
        print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
    
    # Test the model
    test_prediction_mu, test_prediction_sigma_sq = model.apply(parameters, test_inputs)
    
    # Extract arrays from IrrepsArray if needed
    if isinstance(test_prediction_mu, IrrepsArray):
        test_prediction_mu_array = test_prediction_mu.array
        test_prediction_sigma_sq_array = test_prediction_sigma_sq.array
        test_targets_array = test_targets.array if hasattr(test_targets, 'array') else test_targets
    else:
        test_prediction_mu_array = test_prediction_mu
        test_prediction_sigma_sq_array = test_prediction_sigma_sq
        test_targets_array = test_targets
    
    test_mse = jnp.mean((test_prediction_mu_array - test_targets_array) ** 2)
    test_nll = jnp.mean(((test_prediction_mu_array - test_targets_array) ** 2 / (2 * test_prediction_sigma_sq_array)) + 0.5 * jnp.log(test_prediction_sigma_sq_array))
    
    print(f"Test MSE: {test_mse:.4f}, Test NLL: {test_nll:.4f}")
    
    # Save results
    os.makedirs(save_dir, exist_ok=True)
    
    # Convert to numpy arrays for saving
    test_prediction_mu_np = np.array(test_prediction_mu_array)
    test_prediction_sigma_sq_np = np.array(test_prediction_sigma_sq_array)
    test_targets_np = np.array(test_targets_array)
    
    # Get the right test inputs for saving
    if isinstance(test_inputs, IrrepsArray) and hasattr(test_inputs, 'array'):
        test_inputs_np = np.array(test_inputs.array)
    else:
        test_inputs_np = np.array(test_inputs)
    
    # Save predictions, targets and inputs
    np.save(f"{save_dir}/test_prediction_mu_{model_type}.npy", test_prediction_mu_np)
    np.save(f"{save_dir}/test_prediction_sigma_sq_{model_type}.npy", test_prediction_sigma_sq_np)
    np.save(f"{save_dir}/test_targets_{model_type}.npy", test_targets_np)
    np.save(f"{save_dir}/test_inputs_{model_type}.npy", test_inputs_np)
    
    # Save training history
    np.save(f"{save_dir}/train_losses_{model_type}.npy", np.array(train_losses))
    np.save(f"{save_dir}/val_losses_{model_type}.npy", np.array(val_losses))
    
    return {
        'model_type': model_type,
        'parameters': parameters,
        'test_mse': float(test_mse),
        'test_nll': float(test_nll),
        'test_predictions_mu': test_prediction_mu_np,
        'test_predictions_sigma_sq': test_prediction_sigma_sq_np
    }

def test_with_rotations(model, model_type, parameters, test_inputs, test_targets, n_rotations=8, save_dir=None):
    """
    Test the model with rotated inputs to analyze equivariance properties.
    
    Args:
        model: The trained model
        model_type: Type of model ('correct_equivariant', 'incorrect_equivariant', 'mlp')
        parameters: Model parameters
        test_inputs: Test inputs
        test_targets: Test targets
        n_rotations: Number of rotations to test
        save_dir: Directory to save results
    """
    rotation_angles = np.linspace(0, 2*np.pi, n_rotations, endpoint=False)
    mse_results = []
    nll_results = []
    calibration_results = []
    
    # Process inputs based on model type
    is_equivariant = model_type in ['correct_equivariant', 'incorrect_equivariant']
    
    for angle in rotation_angles:
        # Create rotation matrix
        cos_theta = np.cos(angle)
        sin_theta = np.sin(angle)
        rotation_matrix = np.array([
            [cos_theta, -sin_theta],
            [sin_theta, cos_theta]
        ])
        
        if is_equivariant:
            # Extract arrays from IrrepsArray if needed
            if hasattr(test_inputs, 'array'):
                inputs_array = test_inputs.array[:, :2]  # Extract x,y components
                targets_array = test_targets.array[:, :2]  # Extract x,y components
            else:
                inputs_array = test_inputs[:, :2]
                targets_array = test_targets[:, :2]
                
            # Rotate inputs
            rotated_inputs = np.einsum('ij,nj->ni', rotation_matrix, inputs_array)
            
            rotated_inputs_padded = np.pad(rotated_inputs, ((0, 0), (0, 1)), mode="constant")
            rotated_inputs_irreps = IrrepsArray(e3nn.Irreps("1x1e"), rotated_inputs_padded)
            
            # get predictions
            pred_mu, pred_sigma_sq = model.apply(parameters, rotated_inputs_irreps)
            
            # extract arrays from predictions
            if hasattr(pred_mu, 'array'):
                pred_mu_array = pred_mu.array[:, :2]  # Use only x,y components
                pred_sigma_sq_array = pred_sigma_sq.array[:, :2]
            else:
                pred_mu_array = pred_mu[:, :2]
                pred_sigma_sq_array = pred_sigma_sq[:, :2]
            
            # rotate the targets appropriately for evaluation
            rotated_targets = np.einsum('ij,nj->ni', rotation_matrix, targets_array)
            
        else:
            # For MLP, we rotate the inputs and apply the inverse rotation to outputs
            rotated_inputs = np.einsum('ij,nj->ni', rotation_matrix, test_inputs)
            
            # Get predictions
            pred_mu, pred_sigma_sq = model.apply(parameters, rotated_inputs)
            
            # For non-equivariant models, we need to rotate the predictions back
            # to compare with the original targets
            pred_mu_array = np.einsum('ji,nj->ni', rotation_matrix, pred_mu)
            pred_sigma_sq_array = pred_sigma_sq  # Uncertainty doesn't need rotation
            
            # Original targets (no rotation needed)
            rotated_targets = test_targets
        
        # Calculate metrics
        mse = np.mean((pred_mu_array - rotated_targets) ** 2)
        nll = np.mean(((pred_mu_array - rotated_targets) ** 2 / (2 * pred_sigma_sq_array)) + 0.5 * np.log(pred_sigma_sq_array))
        
        # Calculate calibration metric (expected vs actual error)
        expected_error = np.sqrt(pred_sigma_sq_array)
        actual_error = np.abs(pred_mu_array - rotated_targets)
        calibration = np.mean(actual_error / (expected_error + 1e-6))
        
        mse_results.append(mse)
        nll_results.append(nll)
        calibration_results.append(calibration)
    
    # Save results
    if save_dir:
        np.save(f"{save_dir}/rotation_angles_{model_type}.npy", rotation_angles)
        np.save(f"{save_dir}/rotation_mse_{model_type}.npy", np.array(mse_results))
        np.save(f"{save_dir}/rotation_nll_{model_type}.npy", np.array(nll_results))
        np.save(f"{save_dir}/rotation_calibration_{model_type}.npy", np.array(calibration_results))
    
    return {
        'angles': rotation_angles,
        'mse': mse_results,
        'nll': nll_results,
        'calibration': calibration_results
    }

def visualize_predictions(test_inputs, test_targets, predictions_mu, predictions_sigma_sq, title, save_path=None, vmin=None, vmax=None):
    """
    Visualize vector field predictions with uncertainty.
    
    Args:
        test_inputs: Test input positions
        test_targets: True vector field
        predictions_mu: Predicted vector field
        predictions_sigma_sq: Predicted uncertainty
        title: Plot title
        save_path: Path to save the figure
        vmin: Minimum value for uncertainty colormap (for consistent scaling)
        vmax: Maximum value for uncertainty colormap (for consistent scaling)
    """
    # Extract arrays if needed
    if hasattr(test_inputs, 'array'):
        test_inputs_array = test_inputs.array
    else:
        test_inputs_array = test_inputs
        
    if hasattr(test_targets, 'array'):
        test_targets_array = test_targets.array
    else:
        test_targets_array = test_targets
        
    if hasattr(predictions_mu, 'array'):
        predictions_mu_array = predictions_mu.array
        predictions_sigma_sq_array = predictions_sigma_sq.array
    else:
        predictions_mu_array = predictions_mu
        predictions_sigma_sq_array = predictions_sigma_sq
    
    # Extract 2D components if inputs are in 3D
    if test_inputs_array.shape[1] > 2:
        test_inputs_array = test_inputs_array[:, :2]
    if test_targets_array.shape[1] > 2:
        test_targets_array = test_targets_array[:, :2]
    if predictions_mu_array.shape[1] > 2:
        predictions_mu_array = predictions_mu_array[:, :2]
    if predictions_sigma_sq_array.shape[1] > 2:
        predictions_sigma_sq_array = predictions_sigma_sq_array[:, :2]
    
    # Calculate uncertainty as standard deviation
    #predictions_sigma = np.sqrt(predictions_sigma_sq_array)
    predictions_sigma = predictions_sigma_sq_array
    #  uncertainty = np.mean(predictions_sigma, axis=1)
    uncertainty = np.linalg.norm(predictions_sigma, axis=1)
    
    # Create figure
    fig, ax = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot ground truth
    ax[0].quiver(test_inputs_array[:, 0], test_inputs_array[:, 1], 
                test_targets_array[:, 0], test_targets_array[:, 1], 
                angles='xy', scale_units='xy', scale=1, color='blue', alpha=0.8)
    ax[0].set_title('Ground Truth Vector Field')
    ax[0].set_xlabel('X')
    ax[0].set_ylabel('Y')
    ax[0].grid(True)
    ax[0].axis('equal')
    
    # Create a normalizer for consistent color scaling
    if vmin is not None and vmax is not None:
        norm = plt.Normalize(vmin, vmax)
    else:
        norm = plt.Normalize(uncertainty.min(), uncertainty.max())
    
    # Plot predictions with uncertainty
    scatter = ax[1].quiver(test_inputs_array[:, 0], test_inputs_array[:, 1], 
                         predictions_mu_array[:, 0], predictions_mu_array[:, 1], 
                         uncertainty, cmap='viridis', norm=norm,
                         angles='xy', scale_units='xy', scale=1, alpha=0.8)
    cbar = plt.colorbar(scatter, ax=ax[1])
    cbar.set_label(r'Uncertainty $||\sigma^2||$')
    ax[1].set_title(f'Predicted Vector Field with Uncertainty - {title}')
    ax[1].set_xlabel('X')
    ax[1].set_ylabel('Y')
    ax[1].grid(True)
    ax[1].axis('equal')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or show
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def visualize_all_predictions(test_inputs_dict, test_targets_dict, 
                             predictions_mu_dict, predictions_sigma_sq_dict, 
                             test_mse_dict, save_path=None):
    """
    Visualize vector field predictions from all models in one figure with error bar comparison.
    
    Args:
        test_inputs_dict: Dictionary of test input positions for each model type
        test_targets_dict: Dictionary of true vector fields for each model type
        predictions_mu_dict: Dictionary of predicted vector fields for each model type
        predictions_sigma_sq_dict: Dictionary of predicted uncertainties for each model type
        test_mse_dict: Dictionary of test MSE values for each model type
        save_path: Path to save the figure
    """
    # Define model mapping for display purposes
    model_display = {
        'mlp': 'MLP',
        'equivariant': 'E(3)'
    }
    
    # Define the order of models to display
    model_order = ['mlp', 'equivariant']
    
    # Create a figure with 1×4 layout (ground truth + 3 models)
    fig = plt.figure(figsize=(18, 6))
    
    # Use GridSpec to create a 1×4 grid with space for colorbar
    gs = GridSpec(1, 3, figure=fig)
    
    # Ground truth subplot
    ax_truth = fig.add_subplot(gs[0, 0])
    
    # Process and plot each model
    model_axes = []
    all_errors = []  # To collect error data for the error bar plot
    all_uncertainties = []  # To collect all uncertainty values for consistent scaling
    
    # Create axes for each model
    ax_mlp = fig.add_subplot(gs[0, 1])
    model_axes.append((ax_mlp, 'mlp'))
    
    ax_equivariant = fig.add_subplot(gs[0, 2])
    model_axes.append((ax_equivariant, 'equivariant'))
    
    #ax_invariant = fig.add_subplot(gs[0, 3])
    #model_axes.append((ax_invariant, 'invariant'))
    
    # First, get ground truth from any model (they should all have the same ground truth)
    first_model = model_order[0]
    if first_model in test_inputs_dict:
        # Extract arrays if needed
        if hasattr(test_inputs_dict[first_model], 'array'):
            truth_inputs_array = test_inputs_dict[first_model].array
        else:
            truth_inputs_array = test_inputs_dict[first_model]
            
        if hasattr(test_targets_dict[first_model], 'array'):
            truth_targets_array = test_targets_dict[first_model].array
        else:
            truth_targets_array = test_targets_dict[first_model]
        
        # Extract 2D components if inputs are in 3D
        if truth_inputs_array.shape[1] > 2:
            truth_inputs_array = truth_inputs_array[:, :2]
        if truth_targets_array.shape[1] > 2:
            truth_targets_array = truth_targets_array[:, :2]
        
        # Plot ground truth
        ax_truth.quiver(truth_inputs_array[:, 0], truth_inputs_array[:, 1], 
                   truth_targets_array[:, 0], truth_targets_array[:, 1], 
                   angles='xy', scale_units='xy', scale=1, color='blue', alpha=0.8)
        ax_truth.set_title('Ground Truth Vector Field')
        ax_truth.set_xlabel('X')
        ax_truth.set_ylabel('Y', labelpad=-5)  # Bring Y label closer to the axis
        ax_truth.grid(True)
        
        # Set aspect ratio to be equal (square plot)
        ax_truth.set_aspect('equal')
        
        # Set consistent limits for all plots
        min_x, max_x = -5.5, 5.5
        min_y, max_y = -5.5, 5.5
        ax_truth.set_xlim(min_x, max_x)
        ax_truth.set_ylim(min_y, max_y)
    
    # First pass to collect all uncertainty values for consistent scaling
    model_data = {}
    for model_type in model_order:
        if model_type in test_inputs_dict:
            # Extract arrays if needed
            if hasattr(test_inputs_dict[model_type], 'array'):
                inputs_array = test_inputs_dict[model_type].array
            else:
                inputs_array = test_inputs_dict[model_type]
                
            if hasattr(test_targets_dict[model_type], 'array'):
                targets_array = test_targets_dict[model_type].array
            else:
                targets_array = test_targets_dict[model_type]
                
            if hasattr(predictions_mu_dict[model_type], 'array'):
                pred_mu_array = predictions_mu_dict[model_type].array
                pred_sigma_sq_array = predictions_sigma_sq_dict[model_type].array
            else:
                pred_mu_array = predictions_mu_dict[model_type]
                pred_sigma_sq_array = predictions_sigma_sq_dict[model_type]
            
            # Extract 2D components if inputs are in 3D
            if inputs_array.shape[1] > 2:
                inputs_array = inputs_array[:, :2]
            if targets_array.shape[1] > 2:
                targets_array = targets_array[:, :2]
            if pred_mu_array.shape[1] > 2:
                pred_mu_array = pred_mu_array[:, :2]
            if pred_sigma_sq_array.shape[1] > 2:
                pred_sigma_sq_array = pred_sigma_sq_array[:, :2]
            
            # Calculate uncertainty as standard deviation
            #pred_sigma = np.sqrt(pred_sigma_sq_array)
            pred_sigma = pred_sigma_sq_array
            uncertainty = np.linalg.norm(pred_sigma, axis=1)
            all_uncertainties.extend(uncertainty)
            
            # Calculate error for error bar comparison
            error = np.linalg.norm(pred_mu_array - targets_array, axis=1)
            all_errors.append((model_type, error, uncertainty))
            
            # Store processed data for second pass
            model_data[model_type] = {
                'inputs': inputs_array,
                'targets': targets_array,
                'pred_mu': pred_mu_array,
                'uncertainty': uncertainty
            }
    
    # Determine global min and max uncertainty for consistent color scaling
    vmin = min(all_uncertainties) if all_uncertainties else 0
    vmax = max(all_uncertainties) if all_uncertainties else 1
    
    # Create normalizer for consistent color scaling
    norm = plt.Normalize(vmin, vmax)
    
    # Store all quiver plots to create a shared colorbar
    quiver_plots = []
    
    # Set consistent limits for all plots
    min_x, max_x = -5.5, 5.5
    min_y, max_y = -5.5, 5.5
    
    # Second pass to plot models with consistent uncertainty color scaling
    for ax, model_type in model_axes:
        if model_type in model_data:
            data = model_data[model_type]
            
            # Plot predictions with consistent uncertainty scale
            # Use the display name mapping for the title
            display_name = model_display.get(model_type, model_type.capitalize())
            title = f"{display_name} (MSE: {test_mse_dict.get(model_type, 0):.4f})"
            quiver = ax.quiver(data['inputs'][:, 0], data['inputs'][:, 1], 
                              data['pred_mu'][:, 0], data['pred_mu'][:, 1], 
                              data['uncertainty'], cmap='viridis', 
                              angles='xy', scale_units='xy', scale=1, alpha=0.8,
                              norm=norm)  # Use consistent color normalization
            
            quiver_plots.append(quiver)
            
            ax.set_title(title)
            ax.set_xlabel('X')
            ax.set_ylabel('Y', labelpad=-5)  # Bring Y label closer to the axis
            ax.grid(True)
            
            # Set aspect ratio to be equal (square plot)
            ax.set_aspect('equal')
            
            # Apply consistent limits
            ax.set_xlim(min_x, max_x)
            ax.set_ylim(min_y, max_y)
    
    # Create a single colorbar for all plots
    if quiver_plots:
        # Use the last axis (invariant/incorrect model) for the colorbar
        last_ax = model_axes[-1][0]
        
        # Create colorbar with axes_divider for proper alignment
        divider = make_axes_locatable(last_ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        
        # Create the colorbar
        cbar = fig.colorbar(quiver_plots[-1], cax=cax)
        
        # Calculate fewer tick positions for more relaxed indexing
        tick_count = 5  # Reduced from 10 for more relaxed appearance
        ticks = np.linspace(vmin, vmax, tick_count)
        cbar.set_ticks(ticks)
        
        # Format the tick labels with 2 decimal places
        cbar.set_ticklabels([f'{tick:.2f}' for tick in ticks])
        cbar.set_label(r'Uncertainty $||\sigma^2||$')

        second_last_ax = model_axes[-2][0]  # This is the second-to-last axis
        divider2 = make_axes_locatable(second_last_ax)
        cax2 = divider2.append_axes("right", size="5%", pad=0.1)
        cbar2 = fig.colorbar(quiver_plots[-1], cax=cax2)
        cbar2.set_ticks(ticks)
        cbar2.set_ticklabels([f'{tick:.2f}' for tick in ticks])

        inset_ax = inset_axes(last_ax, width="40%", height="40%", loc='upper right', borderpad=1)

        inset_quiver = inset_ax.quiver(
            data['inputs'][:, 0],
            data['inputs'][:, 1],
            100 * data['pred_mu'][:, 0],  # Scale the X component by 100
            100 * data['pred_mu'][:, 1],  # Scale the Y component by 100
            data['uncertainty'],
            cmap='viridis',
            angles='xy',
            scale_units='xy',
            scale=1,  # Keep the scale same so that the multiplication affects the arrow length
            alpha=0.8,
            norm=norm
        )
    
        # Optionally, customize the inset axes (e.g., add a title and adjust limits)
        inset_ax.set_title("Scaled x100", fontsize=8)
        inset_ax.set_xlim(min_x, max_x)
        inset_ax.set_ylim(min_y, max_y)
        inset_ax.grid(True)
    
    # Create error bar comparison in a separate figure
    if all_errors:
        # Add error bar plot in a separate figure to maintain clean layout
        fig_error = plt.figure(figsize=(10, 8))
        ax_error = fig_error.add_subplot(111)
        
        # Sort points by error for clearer visualization
        for model_type, error, uncertainty in all_errors:
            # Create sorted indices
            sorted_indices = np.argsort(error)
            sorted_error = error[sorted_indices]
            sorted_uncertainty = uncertainty[sorted_indices]
            
            # Plot error vs predicted uncertainty - use display name
            display_name = model_display.get(model_type, model_type.capitalize())
            ax_error.scatter(sorted_error, sorted_uncertainty, label=display_name, alpha=0.7)
        
        # Add diagonal line for perfect calibration
        max_val = max(np.max([e for _, e, _ in all_errors]), np.max([u for _, _, u in all_errors]))
        ax_error.plot([0, max_val], [0, max_val], 'k--', label='Perfect Calibration')
        
        ax_error.set_title('Error vs. Predicted Uncertainty')
        ax_error.set_xlabel('Actual Error')
        ax_error.set_ylabel('Predicted Uncertainty')
        ax_error.legend()
        ax_error.grid(True)
        
        # Save error plot separately
        if save_path:
            error_save_path = save_path.replace('.pdf', '_error_comparison.pdf')
            fig_error.tight_layout()
            fig_error.savefig(error_save_path)
            plt.close(fig_error)
    
    # Adjust layout 
    plt.tight_layout()
    
    # Save or show
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_rotation_results(rotation_results, save_dir=None):
    """
    Plot results of rotation tests.
    
    Args:
        rotation_results: Dictionary of rotation test results for different models
        save_dir: Directory to save results
    """
    model_types = list(rotation_results.keys())
    fig, ax = plt.subplots(1, 3, figsize=(24, 6))
    
    # Extract angles (they should be the same for all models)
    angles_deg = np.degrees(rotation_results[model_types[0]]['angles'])
    
    # Plot MSE
    for model_type in model_types:
        ax[0].plot(angles_deg, rotation_results[model_type]['mse'], 
                  label=model_type, marker='o')
    ax[0].set_title('MSE vs. Rotation Angle')
    ax[0].set_xlabel('Rotation Angle (degrees)')
    ax[0].set_ylabel('Mean Squared Error')
    ax[0].legend()
    ax[0].grid(True)
    
    # Plot NLL
    for model_type in model_types:
        ax[1].plot(angles_deg, rotation_results[model_type]['nll'], 
                  label=model_type, marker='o')
    ax[1].set_title('NLL vs. Rotation Angle')
    ax[1].set_xlabel('Rotation Angle (degrees)')
    ax[1].set_ylabel('Negative Log-Likelihood')
    ax[1].legend()
    ax[1].grid(True)
    
    # Plot Calibration
    for model_type in model_types:
        ax[2].plot(angles_deg, rotation_results[model_type]['calibration'], 
                  label=model_type, marker='o')
    ax[2].axhline(y=1.0, color='r', linestyle='--', label='Perfect Calibration')
    ax[2].set_title('Uncertainty Calibration vs. Rotation Angle')
    ax[2].set_xlabel('Rotation Angle (degrees)')
    ax[2].set_ylabel('Actual Error / Expected Error')
    ax[2].legend()
    ax[2].grid(True)
    
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(f"{save_dir}/rotation_results_comparison.pdf")
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    parser = ArgumentParser()
    
    @dataclass
    class Options:
        dataset_type: str = "spiral"
        n_samples: int = 2000
        noise: float = 0.1
        hidden_dim: int = 32
        batch_size: int = 64
        beta: float = 1.0
        n_holdout: int = 200
        minimum_epochs: int = 10
        maximum_epochs_no_improve: int = 10
        maximum_epochs: int = 100
        train_val_split: float = 0.8
        n_rotations: int = 8
        data_dir: str = "../data/vector_field"
        results_dir: str = "../results/vector_field"
        random_seed: int = 42
        skip_training: bool = False  # Flag to skip training and just run visualizations
    
    parser.add_arguments(Options, dest="options")
    args = parser.parse_args()
    
    # Extract arguments
    dataset_type = args.options.dataset_type
    n_samples = args.options.n_samples
    noise = args.options.noise
    hidden_dim = args.options.hidden_dim
    batch_size = args.options.batch_size
    beta = args.options.beta
    n_holdout = args.options.n_holdout
    minimum_epochs = args.options.minimum_epochs
    maximum_epochs_no_improve = args.options.maximum_epochs_no_improve
    maximum_epochs = args.options.maximum_epochs
    train_val_split = args.options.train_val_split
    n_rotations = args.options.n_rotations
    data_dir = args.options.data_dir
    results_dir = args.options.results_dir
    random_seed = args.options.random_seed
    skip_training = args.options.skip_training
    
    # Set random seed
    key = jax.random.PRNGKey(random_seed)
    set_rc_params(fontsize=28)
    
    # Create directories
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Load or generate dataset
    data_file_prefix = f"{data_dir}/input_positions_{dataset_type}_{n_samples}_samples_{noise}_noise"
    if os.path.exists(f"{data_file_prefix}.npy"):
        print(f"Loading existing dataset from {data_file_prefix}")
        input_positions = np.load(f"{data_file_prefix}.npy")
        vector_field = np.load(f"{data_dir}/vector_field_{dataset_type}_{n_samples}_samples_{noise}_noise.npy")
    else:
        print(f"Generating new dataset of type {dataset_type}")
        # Import the dataset creation function from the make_vector_field_dataset.py file
        from make_vector_field_dataset import create_synthetic_vector_field
        input_positions, vector_field = create_synthetic_vector_field(n_samples, noise, dataset_type)
        
        # Save the dataset
        np.save(f"{data_file_prefix}.npy", input_positions)
        np.save(f"{data_dir}/vector_field_{dataset_type}_{n_samples}_samples_{noise}_noise.npy", vector_field)
    
    # Define file paths for saved model results
    correct_model_files = {
        'pred_mu': f"{results_dir}/test_prediction_mu_correct_equivariant.npy",
        'pred_sigma_sq': f"{results_dir}/test_prediction_sigma_sq_correct_equivariant.npy",
        'test_targets': f"{results_dir}/test_targets_correct_equivariant.npy",
        'test_inputs': f"{results_dir}/test_inputs_correct_equivariant.npy",
        'rotation_mse': f"{results_dir}/rotation_mse_correct_equivariant.npy",
        'rotation_nll': f"{results_dir}/rotation_nll_correct_equivariant.npy",
        'rotation_calibration': f"{results_dir}/rotation_calibration_correct_equivariant.npy",
        'rotation_angles': f"{results_dir}/rotation_angles_correct_equivariant.npy"
    }
    
    incorrect_model_files = {
        'pred_mu': f"{results_dir}/test_prediction_mu_incorrect_equivariant.npy",
        'pred_sigma_sq': f"{results_dir}/test_prediction_sigma_sq_incorrect_equivariant.npy",
        'test_targets': f"{results_dir}/test_targets_incorrect_equivariant.npy",
        'test_inputs': f"{results_dir}/test_inputs_incorrect_equivariant.npy",
        'rotation_mse': f"{results_dir}/rotation_mse_incorrect_equivariant.npy",
        'rotation_nll': f"{results_dir}/rotation_nll_incorrect_equivariant.npy",
        'rotation_calibration': f"{results_dir}/rotation_calibration_incorrect_equivariant.npy",
        'rotation_angles': f"{results_dir}/rotation_angles_incorrect_equivariant.npy"
    }
    
    mlp_model_files = {
        'pred_mu': f"{results_dir}/test_prediction_mu_mlp.npy",
        'pred_sigma_sq': f"{results_dir}/test_prediction_sigma_sq_mlp.npy",
        'test_targets': f"{results_dir}/test_targets_mlp.npy",
        'test_inputs': f"{results_dir}/test_inputs_mlp.npy",
        'rotation_mse': f"{results_dir}/rotation_mse_mlp.npy",
        'rotation_nll': f"{results_dir}/rotation_nll_mlp.npy",
        'rotation_calibration': f"{results_dir}/rotation_calibration_mlp.npy",
        'rotation_angles': f"{results_dir}/rotation_angles_mlp.npy"
    }
    
    # Check if all necessary files exist to skip training
    all_files_exist = all(os.path.exists(f) for f in 
                         list(correct_model_files.values()) + 
                         list(incorrect_model_files.values()) + 
                         list(mlp_model_files.values()))
    
    # Also check for MSE and NLL files specifically
    mse_nll_files = [
        f"{results_dir}/test_mse_correct_equivariant.npy",
        f"{results_dir}/test_nll_correct_equivariant.npy",
        f"{results_dir}/test_mse_incorrect_equivariant.npy",
        f"{results_dir}/test_nll_incorrect_equivariant.npy",
        f"{results_dir}/test_mse_mlp.npy",
        f"{results_dir}/test_nll_mlp.npy"
    ]
    
    mse_nll_exist = all(os.path.exists(f) for f in mse_nll_files)
    
    # Only skip training if ALL required files exist
    should_train = not skip_training and (not all_files_exist or not mse_nll_exist)
    
    # Initialize result dictionaries
    correct_results = {}
    incorrect_results = {}
    mlp_results = {}
    
    # Training section (can be commented out or skipped with --skip_training flag)
    if should_train:
        print("\n--- Running model training and evaluation ---")
        
        # Setup for models
        input_irreps = e3nn.Irreps("1x1e")  # 3D vector input (x,y,z with z=0)
        output_irreps = e3nn.Irreps("1x1e")  # 3D vector output (x,y,z with z=0)
        
        key, subkey1, subkey2, subkey3 = jax.random.split(key, 4)
        
        # Create and train correct equivariant model
        correct_model = CorrectEquivariantVectorFieldModel(
            input_irreps=input_irreps,
            output_irreps=output_irreps,
            hidden_dim=hidden_dim
        )
        
        print("\n--- Training Correct Equivariant Model ---")
        correct_results = train_model(
            correct_model, 'correct_equivariant', subkey1, input_positions, vector_field,
            n_holdout, minimum_epochs, maximum_epochs_no_improve, maximum_epochs,
            batch_size, train_val_split, beta, results_dir
        )
        
        # Create and train incorrect equivariant model
        incorrect_model = IncorrectEquivariantVectorFieldModel(
            input_irreps=input_irreps,
            output_irreps=output_irreps,
            hidden_dim=hidden_dim
        )
        
        print("\n--- Training Incorrect Equivariant Model ---")
        incorrect_results = train_model(
            incorrect_model, 'incorrect_equivariant', subkey2, input_positions, vector_field,
            n_holdout, minimum_epochs, maximum_epochs_no_improve, maximum_epochs,
            batch_size, train_val_split, beta, results_dir
        )
        
        # Create and train MLP model
        mlp_model = MLPVectorFieldModel(
            hidden_dim=hidden_dim,
            output_dim=2  # 2D vector output (x,y)
        )
        
        print("\n--- Training MLP Model ---")
        mlp_results = train_model(
            mlp_model, 'mlp', subkey3, input_positions, vector_field,
            n_holdout, minimum_epochs, maximum_epochs_no_improve, maximum_epochs,
            batch_size, train_val_split, beta, results_dir
        )
        
        # Testing with rotations
        print("\n--- Testing Models with Rotations ---")
        
        # Prepare test data for rotation tests
        test_indices = np.arange(n_samples - n_holdout, n_samples)
        
        # Test data for each model type
        correct_test_inputs = np.load(f"{results_dir}/test_inputs_correct_equivariant.npy")
        correct_test_targets = np.load(f"{results_dir}/test_targets_correct_equivariant.npy")
        
        incorrect_test_inputs = np.load(f"{results_dir}/test_inputs_incorrect_equivariant.npy")
        incorrect_test_targets = np.load(f"{results_dir}/test_targets_incorrect_equivariant.npy")
        
        mlp_test_inputs = np.load(f"{results_dir}/test_inputs_mlp.npy")
        mlp_test_targets = np.load(f"{results_dir}/test_targets_mlp.npy")
        
        # Convert to IrrepsArray for equivariant models
        correct_test_inputs_irreps = IrrepsArray(input_irreps, correct_test_inputs)
        correct_test_targets_irreps = IrrepsArray(output_irreps, correct_test_targets)
        
        incorrect_test_inputs_irreps = IrrepsArray(input_irreps, incorrect_test_inputs)
        incorrect_test_targets_irreps = IrrepsArray(output_irreps, incorrect_test_targets)
        
        # Perform rotation tests
        correct_rotation_results = test_with_rotations(
            correct_model, 'correct_equivariant', 
            correct_results['parameters'],
            correct_test_inputs_irreps, correct_test_targets_irreps,
            n_rotations, results_dir
        )
        
        incorrect_rotation_results = test_with_rotations(
            incorrect_model, 'incorrect_equivariant', 
            incorrect_results['parameters'],
            incorrect_test_inputs_irreps, incorrect_test_targets_irreps,
            n_rotations, results_dir
        )
        
        mlp_rotation_results = test_with_rotations(
            mlp_model, 'mlp', 
            mlp_results['parameters'],
            mlp_test_inputs, mlp_test_targets,
            n_rotations, results_dir
        )
        
        # Save MSE and NLL values for later use
        np.save(f"{results_dir}/test_mse_correct_equivariant.npy", correct_results['test_mse'])
        np.save(f"{results_dir}/test_nll_correct_equivariant.npy", correct_results['test_nll'])
        
        np.save(f"{results_dir}/test_mse_incorrect_equivariant.npy", incorrect_results['test_mse'])
        np.save(f"{results_dir}/test_nll_incorrect_equivariant.npy", incorrect_results['test_nll'])
        
        np.save(f"{results_dir}/test_mse_mlp.npy", mlp_results['test_mse'])
        np.save(f"{results_dir}/test_nll_mlp.npy", mlp_results['test_nll'])
    else:
        print("\n--- Skipping training, using existing model results ---")
    
    # Visualization section (runs regardless of whether training was skipped)
    print("\n--- Creating Visualizations ---")
    
    # Define a helper function to safely load NumPy files
    def safe_load(file_path, default=None):
        try:
            return np.load(file_path)
        except FileNotFoundError:
            print(f"Warning: Could not load {file_path}")
            if default is not None:
                print(f"Using default value instead")
                return default
            else:
                raise ValueError(f"Required file {file_path} not found. Run training first.")
    
    # Load all necessary files for visualization, with error handling
    try:
        # Load model predictions and test data
        correct_pred_mu = safe_load(correct_model_files['pred_mu'])
        correct_pred_sigma_sq = safe_load(correct_model_files['pred_sigma_sq'])
        correct_test_targets = safe_load(correct_model_files['test_targets'])
        correct_test_inputs = safe_load(correct_model_files['test_inputs'])
        
        incorrect_pred_mu = safe_load(incorrect_model_files['pred_mu'])
        incorrect_pred_sigma_sq = safe_load(incorrect_model_files['pred_sigma_sq'])
        incorrect_test_targets = safe_load(incorrect_model_files['test_targets'])
        incorrect_test_inputs = safe_load(incorrect_model_files['test_inputs'])
        
        mlp_pred_mu = safe_load(mlp_model_files['pred_mu'])
        mlp_pred_sigma_sq = safe_load(mlp_model_files['pred_sigma_sq'])
        mlp_test_targets = safe_load(mlp_model_files['test_targets'])
        mlp_test_inputs = safe_load(mlp_model_files['test_inputs'])
        
        # Load rotation results
        correct_rotation_angles = safe_load(correct_model_files['rotation_angles'])
        correct_rotation_mse = safe_load(correct_model_files['rotation_mse'])
        correct_rotation_nll = safe_load(correct_model_files['rotation_nll'])
        correct_rotation_calibration = safe_load(correct_model_files['rotation_calibration'])
        
        incorrect_rotation_angles = safe_load(incorrect_model_files['rotation_angles'])
        incorrect_rotation_mse = safe_load(incorrect_model_files['rotation_mse'])
        incorrect_rotation_nll = safe_load(incorrect_model_files['rotation_nll'])
        incorrect_rotation_calibration = safe_load(incorrect_model_files['rotation_calibration'])
        
        mlp_rotation_angles = safe_load(mlp_model_files['rotation_angles'])
        mlp_rotation_mse = safe_load(mlp_model_files['rotation_mse'])
        mlp_rotation_nll = safe_load(mlp_model_files['rotation_nll'])
        mlp_rotation_calibration = safe_load(mlp_model_files['rotation_calibration'])
        
        # Load MSE and NLL values if needed
        if not correct_results:
            correct_results = {
                'test_mse': float(safe_load(f"{results_dir}/test_mse_correct_equivariant.npy")),
                'test_nll': float(safe_load(f"{results_dir}/test_nll_correct_equivariant.npy"))
            }
        
        if not incorrect_results:
            incorrect_results = {
                'test_mse': float(safe_load(f"{results_dir}/test_mse_incorrect_equivariant.npy")),
                'test_nll': float(safe_load(f"{results_dir}/test_nll_incorrect_equivariant.npy"))
            }
        
        if not mlp_results:
            mlp_results = {
                'test_mse': float(safe_load(f"{results_dir}/test_mse_mlp.npy")),
                'test_nll': float(safe_load(f"{results_dir}/test_nll_mlp.npy"))
            }
        
        # Reconstruct rotation results dictionaries for plotting
        correct_rotation_results = {
            'angles': correct_rotation_angles,
            'mse': correct_rotation_mse,
            'nll': correct_rotation_nll,
            'calibration': correct_rotation_calibration
        }
        
        incorrect_rotation_results = {
            'angles': incorrect_rotation_angles,
            'mse': incorrect_rotation_mse,
            'nll': incorrect_rotation_nll,
            'calibration': incorrect_rotation_calibration
        }
        
        mlp_rotation_results = {
            'angles': mlp_rotation_angles,
            'mse': mlp_rotation_mse,
            'nll': mlp_rotation_nll,
            'calibration': mlp_rotation_calibration
        }
        
        # Visualize individual model predictions
        
        # Calculate all uncertainties to get consistent color scale
        all_uncertainties = []
        
        # Collect uncertainties from all models
        for model_type in ['mlp', 'equivariant', 'invariant']:
            if model_type == 'mlp':
                pred_sigma_sq = mlp_pred_sigma_sq
            elif model_type == 'equivariant':
                pred_sigma_sq = correct_pred_sigma_sq
            elif model_type == 'invariant':
                pred_sigma_sq = incorrect_pred_sigma_sq
                
            # Extract arrays if needed
            if hasattr(pred_sigma_sq, 'array'):
                pred_sigma_sq_array = pred_sigma_sq.array
            else:
                pred_sigma_sq_array = pred_sigma_sq
                
            # Handle 3D arrays by using only x,y components
            if pred_sigma_sq_array.shape[1] > 2:
                pred_sigma_sq_array = pred_sigma_sq_array[:, :2]
                
            # Calculate uncertainty
            pred_sigma = np.sqrt(pred_sigma_sq_array)
            uncertainty = np.linalg.norm(pred_sigma, axis=1)
            all_uncertainties.extend(uncertainty)
        
        # Determine global min and max for uncertainty
        vmin = min(all_uncertainties) if all_uncertainties else 0
        vmax = max(all_uncertainties) if all_uncertainties else 1
        
        # Create visualizations with consistent color scale
        visualize_predictions(
            correct_test_inputs, correct_test_targets,
            correct_pred_mu, correct_pred_sigma_sq,
            f"Correct Equivariant Model (MSE: {correct_results['test_mse']:.4f})",
            f"{results_dir}/correct_equivariant_predictions.pdf",
            vmin=vmin, vmax=vmax
        )
        
        visualize_predictions(
            incorrect_test_inputs, incorrect_test_targets,
            incorrect_pred_mu, incorrect_pred_sigma_sq,
            f"Incorrect Equivariant Model (MSE: {incorrect_results['test_mse']:.4f})",
            f"{results_dir}/incorrect_equivariant_predictions.pdf",
            vmin=vmin, vmax=vmax
        )
        
        visualize_predictions(
            mlp_test_inputs, mlp_test_targets,
            mlp_pred_mu, mlp_pred_sigma_sq,
            f"MLP Model (MSE: {mlp_results['test_mse']:.4f})",
            f"{results_dir}/mlp_predictions.pdf",
            vmin=vmin, vmax=vmax
        )
        
        # Plot rotation test results
        rotation_results = {
            'Correct Equivariant': correct_rotation_results,
            'MLP': mlp_rotation_results
        }
        
        plot_rotation_results(rotation_results, results_dir)
        
        # Combined visualization of all models
        test_inputs_dict = {
            'mlp': mlp_test_inputs,
            'equivariant': correct_test_inputs,
            'invariant': incorrect_test_inputs  # Using incorrect equivariant as invariant for this example
        }
        
        test_targets_dict = {
            'mlp': mlp_test_targets,
            'equivariant': correct_test_targets,
            'invariant': incorrect_test_targets
        }
        
        predictions_mu_dict = {
            'mlp': mlp_pred_mu,
            'equivariant': correct_pred_mu,
            'invariant': incorrect_pred_mu
        }
        
        predictions_sigma_sq_dict = {
            'mlp': mlp_pred_sigma_sq,
            'equivariant': correct_pred_sigma_sq,
            'invariant': incorrect_pred_sigma_sq
        }
        
        # Create a dictionary of test MSE values
        test_mse_dict = {
            'mlp': mlp_results['test_mse'],
            'equivariant': correct_results['test_mse'],
            'invariant': incorrect_results['test_mse']
        }
        
        # Create the combined visualization
        visualize_all_predictions(
            test_inputs_dict, test_targets_dict,
            predictions_mu_dict, predictions_sigma_sq_dict,
            test_mse_dict,
            f"{results_dir}/all_model_predictions_comparison.pdf"
        )
        
        print(f"\nExperiment completed. Results saved to {results_dir}")
        print("\nTest MSE Results:")
        print(f"  Correct Equivariant: {correct_results['test_mse']:.4f}")
        print(f"  Incorrect Equivariant: {incorrect_results['test_mse']:.4f}")
        print(f"  MLP: {mlp_results['test_mse']:.4f}")
        
        print("\nTest NLL Results:")
        print(f"  Correct Equivariant: {correct_results['test_nll']:.4f}")
        print(f"  Incorrect Equivariant: {incorrect_results['test_nll']:.4f}")
        print(f"  MLP: {mlp_results['test_nll']:.4f}")
    
    except ValueError as e:
        print(f"\nError: {e}")
        print("Please run the script without the --skip_training flag to generate the necessary files.") 

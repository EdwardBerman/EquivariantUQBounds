import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import os
from dataclasses import dataclass
from simple_parsing import ArgumentParser

from matplotlib import rcParams, rc
from matplotlib.gridspec import GridSpec
import matplotlib.patches as patches
import networkx as nx

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

set_rc_params(fontsize=28)


def create_synthetic_vector_field(n_samples, noise_level, dataset_type="spiral"):
    """
    Create a synthetic vector field dataset that follows specific equivariant properties.
    
    Args:
        n_samples: Number of data points to generate
        noise_level: Amount of noise to add to the vector field
        dataset_type: Type of vector field ('spiral', 'rotational', 'divergent')
        
    Returns:
        input_positions: Array of input positions (x,y)
        vector_field: Array of vector field values at each position
    """
    # generate input positions (uniform grid in 2D)
    if dataset_type == "grid":
        # create a grid of points
        side = int(np.sqrt(n_samples))
        x = np.linspace(-5, 5, side)
        y = np.linspace(-5, 5, side)
        xx, yy = np.meshgrid(x, y)
        input_positions = np.column_stack([xx.flatten(), yy.flatten()])
        n_samples = input_positions.shape[0]  # update in case we rounded
    else:
        # use random points
        input_positions = np.random.uniform(-5, 5, size=(n_samples, 2))
    
    # generate vector field based on the type
    if dataset_type == "spiral":
        # siral vector field (rotational + radial)
        vector_field = np.zeros_like(input_positions)
        for i, pos in enumerate(input_positions):
            x, y = pos
            r = np.sqrt(x**2 + y**2) + 1e-6  # Avoid division by zero
            vector_field[i, 0] = -y/r - 0.3*x/r
            vector_field[i, 1] = x/r - 0.3*y/r

    if dataset_type == "spiral-fixed":
        # siral vector field (rotational + radial)
        rot_matrix = np.matrix([[0,1], [-1,0]])
        vector_field = np.zeros_like(input_positions)
        for i, pos in enumerate(input_positions):
            x, y = pos
            #  r = np.sqrt(x**2 + y**2) + 1e-6  # Avoid division by zero
            r = 1
            vector_field[i, 0] = -y/r - 0.3*x/r
            vector_field[i, 1] = x/r - 0.3*y/r
            arc = np.array([vector_field[i, 0],  vector_field[i, 1]])
            res = np.dot(rot_matrix, arc)
            #  print(np.array(res)[0][0])
            vector_field[i, 0] = np.array(res)[0][0]
            vector_field[i, 1] = np.array(res)[0][1]

            #  vector_field[i, 0] = np.dot(rot_matrix, vector_field[i,0])
            #  vector_field[i, 1] = np.dot(rot_matrix, vector_field[i,1])
    
    elif dataset_type == "rotational":
        vector_field = np.zeros_like(input_positions)
        for i, pos in enumerate(input_positions):
            x, y = pos
            vector_field[i, 0] = -y
            vector_field[i, 1] = x
    elif dataset_type == "sine":
        vector_field = np.zeros_like(input_positions)
        for i, pos in enumerate(input_positions):
            x, y = pos
            x_y_vec = np.array([x, y])
            norm = np.linalg.norm(x_y_vec)
            sin_norm = np.sin(norm)
            vector_field[i, 0] = -x*(sin_norm**2)
            vector_field[i, 1] = -y*(sin_norm**2)
    
    elif dataset_type == "divergent":
        # divergent/convergent field (equivariant under rotations)
        vector_field = np.zeros_like(input_positions)
        for i, pos in enumerate(input_positions):
            x, y = pos
            r = np.sqrt(x**2 + y**2) + 1e-6
            # radial field
            vector_field[i, 0] = x/r
            vector_field[i, 1] = y/r
    
    # add noise to the vector field
    if noise_level > 0:
        noise = np.random.normal(0, noise_level, size=vector_field.shape)
        vector_field += noise
    
    return input_positions, vector_field

def plot_vector_field(positions, vectors, title="Vector Field", save_path=None):
    """Plot a vector field for visualization"""
    plt.figure(figsize=(10, 8))
    plt.quiver(positions[:, 0], positions[:, 1], vectors[:, 0], vectors[:, 1], 
               angles='xy', scale_units='xy', scale=1, color='blue', alpha=0.8)
    plt.grid(True)
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axis('equal')
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

if __name__ == '__main__':
    parser = ArgumentParser()

    @dataclass
    class Options:
        n_samples: int = 2000
        noise: float = 0.1
        dataset_type: str = "spiral"  # "spiral", "rotational", "divergent"
        visualize: bool = True
        save_dir: str = "../data/vector_field"

    parser.add_arguments(Options, dest="options")
    args = parser.parse_args()
    
    n_samples = args.options.n_samples
    noise = args.options.noise
    dataset_type = args.options.dataset_type
    visualize = args.options.visualize
    save_dir = args.options.save_dir
    
    # create dataset
    input_positions, vector_field = create_synthetic_vector_field(
        n_samples, noise, dataset_type)
    
    # create directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Save dataset
    np.save(f"{save_dir}/input_positions_{dataset_type}_{n_samples}_samples_{noise}_noise.npy", 
            input_positions)
    np.save(f"{save_dir}/vector_field_{dataset_type}_{n_samples}_samples_{noise}_noise.npy", 
            vector_field)
    
    if visualize:
        plot_vector_field(
            input_positions, vector_field, 
            title=f"{dataset_type.capitalize()} Vector Field (Noise: {noise})",
            save_path=f"{save_dir}/{dataset_type}_vector_field_visualization.png")
    
    print(f"Dataset created and saved to {save_dir}") 

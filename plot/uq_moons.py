import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from matplotlib import rcParams, rc
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Ellipse

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

def plot_moons(input_locations, true_output_locations, output_location_means, output_location_vars, input_locations_equivariant, equivariant_true_output_locations, equivariant_output_location_means, equivariant_output_location_vars):
    fig, ax = plt.subplots(1, 4, figsize=(48, 8))
    ax[0].scatter(input_locations[:, 0], input_locations[:, 1])
    ax[0].set_title("Input")
    ax[0].set_xlabel("x1")
    ax[0].set_ylabel("x2")
    x_axis = np.linspace(-1.5, 2.5, 100)
    y_axis = np.linspace(-1, 1.5, 100)
    ax[0].plot(x_axis, np.zeros_like(x_axis), c="k", linestyle="--")
    ax[0].plot(np.zeros_like(y_axis), y_axis, c="k", linestyle="--")
    ax[1].scatter(true_output_locations[:, 0], true_output_locations[:, 1], c="r", label="True", alpha=0.5)
    ax[1].scatter(output_location_means[:, 0], output_location_means[:, 1], c="b", label="Predicted", alpha=0.5)
    sigma_sq_x = output_location_vars[:, 0]
    sigma_sq_y = output_location_vars[:, 1]
    mean_sigma_norm = np.mean(np.linalg.norm(output_location_vars, axis=1))

    for xi, yi, xe, ye in zip(output_location_means[:, 0], output_location_means[:, 1], sigma_sq_x, sigma_sq_y):
        ellipse = Ellipse(xy=(xi, yi), 
                          width=2*xe, 
                          height=2*ye, 
                          edgecolor="b", 
                          facecolor="None", 
                          linestyle="--",
                          alpha=0.7)
        ax[1].add_patch(ellipse)
    ax[1].set_title(f"MLP Output, AB = {mean_sigma_norm:.2f}")
    ax[1].set_xlabel("y1")
    ax[1].set_ylabel("y2")
    x_axis = np.linspace(-1.5, 2.5, 100)
    y_axis = np.linspace(-1, 1.5, 100)
    ax[1].plot(x_axis, np.zeros_like(x_axis), c="k", linestyle="--")
    ax[1].plot(np.zeros_like(y_axis), y_axis, c="k", linestyle="--")
    ax[1].legend()

    ax[2].scatter(input_locations_equivariant[:, 0], input_locations_equivariant[:, 1])
    ax[2].set_title("Input")
    ax[2].set_xlabel("x1")
    ax[2].set_ylabel("x2")
    x_axis = np.linspace(-1.5, 2.5, 100)
    y_axis = np.linspace(-1, 1.5, 100)
    ax[2].plot(x_axis, np.zeros_like(x_axis), c="k", linestyle="--")
    ax[2].plot(np.zeros_like(y_axis), y_axis, c="k", linestyle="--")

    ax[3].scatter(equivariant_true_output_locations[:, 0], equivariant_true_output_locations[:, 1], c="r", label="True", alpha=0.5)
    ax[3].scatter(equivariant_output_location_means[:, 0], equivariant_output_location_means[:, 1], c="b", label="Predicted", alpha=0.5)
    sigma_sq_x = equivariant_output_location_vars[:, 0]
    sigma_sq_y = equivariant_output_location_vars[:, 1]

    mean_sigma_norm = np.mean(np.linalg.norm(equivariant_output_location_vars, axis=1))

    for xi, yi, xe, ye in zip(equivariant_output_location_means[:, 0], equivariant_output_location_means[:, 1], sigma_sq_x, sigma_sq_y):
        ellipse = Ellipse(xy=(xi, yi), 
                          width=2*xe, 
                          height=2*ye, 
                          edgecolor="b", 
                          facecolor="None", 
                          linestyle="--",
                          alpha=0.7)
        ax[3].add_patch(ellipse)
    ax[3].set_title(f"Equivariant Output, AB = {mean_sigma_norm:.2f}")
    ax[3].set_xlabel("y1")
    ax[3].set_ylabel("y2")
    ax[3].legend()
    x_axis = np.linspace(-1.5, 2.5, 100)
    y_axis = np.linspace(-1, 1.5, 100)
    ax[3].plot(x_axis, np.zeros_like(x_axis), c="k", linestyle="--")
    ax[3].plot(np.zeros_like(y_axis), y_axis, c="k", linestyle="--")
    plt.tight_layout()
    buffer = 0.05
    plt.subplots_adjust(left=buffer, right=1-buffer)
    plt.savefig("../assets/uq_moons.pdf")

if __name__ == "__main__":
    set_rc_params(fontsize=36)

    input_locations_mlp = np.load("../data/test_inputs_MLP.npy")
    true_output_locations_mlp = np.load("../data/test_target_MLP.npy")
    output_location_means_mlp = np.load("../data/test_prediction_mu_MLP.npy")
    output_location_vars_mlp = np.load("../data/test_prediction_sigma_sq_MLP.npy")

    input_locations_equivariant = np.load("../data/test_inputs_equivariant.npy")
    true_output_locations_equivariant = np.load("../data/test_target_equivariant.npy")
    output_location_means_equivariant = np.load("../data/test_prediction_mu_equivariant.npy")
    output_location_vars_equivariant = np.load("../data/test_prediction_sigma_sq_equivariant.npy")

    plot_moons(input_locations_mlp, 
               true_output_locations_mlp, 
               output_location_means_mlp, 
               output_location_vars_mlp, 
               input_locations_equivariant, 
               true_output_locations_equivariant, 
               output_location_means_equivariant, 
               output_location_vars_equivariant)

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from matplotlib import rcParams, rc
from matplotlib.gridspec import GridSpec

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

def plot_moons(input_locations, output_locations):
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    ax[0].scatter(input_locations[:, 0], input_locations[:, 1])
    ax[0].set_title("Input")
    ax[0].set_xlabel("x1")
    ax[0].set_ylabel("x2")
    ax[1].scatter(output_locations[:, 0], output_locations[:, 1])
    ax[1].set_title("Output")
    ax[1].set_xlabel("y1")
    ax[1].set_ylabel("y2")
    plt.tight_layout()
    plt.savefig("../assets/moons.png")

if __name__ == "__main__":
    set_rc_params(fontsize=28)
    input_locations = np.load("../data/input_locations_90_degree_offset_1000_samples_0.1_noise.npy")
    output_locations = np.load("../data/output_locations_90_degree_offset_1000_samples_0.1_noise.npy")
    plot_moons(input_locations, output_locations)

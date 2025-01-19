import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from matplotlib import rcParams, rc

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

    return

set_rc_params(fontsize=16)

# Define parameters for the Gaussians
sigma = 1
mean1 = -3 * sigma
mean2 = 3 * sigma
mean3 = 0  # Middle Gaussian

# Generate x values
x = np.linspace(-10, 10, 1000)

# Compute the original Gaussians
gaussian1 = norm.pdf(x, loc=mean1, scale=sigma)
gaussian2 = norm.pdf(x, loc=mean2, scale=sigma)

# Renormalize to create a single PDF
combined_pdf = (gaussian1 + gaussian2) / (np.trapz(gaussian1 + gaussian2, x))

# Define the third Gaussian
gaussian3 = norm.pdf(x, loc=mean3, scale=sigma)

# Plot the Gaussians
plt.figure(figsize=(10, 8))
plt.plot(x, combined_pdf, label="True Distribution", linewidth=8, color="green")
plt.plot(x, gaussian3, label="Learned Gaussian", linestyle="--", linewidth=8, color="blue")
plt.xlabel("x", fontsize=28)
plt.ylabel("P(x)", fontsize=28)
plt.legend()
plt.grid()
plt.savefig("assets/gaussian_mixture.pdf")


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

set_rc_params(fontsize=28)

bounds = 0, 1
x = np.linspace(bounds[0], bounds[1], 1000)
f_one = np.ones_like(x)
f_two = np.zeros_like(x)
f_three = x

plt.figure(figsize=(10, 10))
plt.plot(x, f_one, label=r"$y = \mathbb{P}(f_Y(x) = h_Y(x) | h_P(x) = p) = 1$", linestyle="--", linewidth=8, color="black")
plt.plot(x, f_two, label=r"$y = \mathbb{P}(f_Y(x) = h_Y(x) | h_P(x) = p) = 0$", linestyle="--", linewidth=8, color="black")
plt.plot(x, f_three, label=r"$y = p$", linestyle="-", linewidth=8, color="orange")
plt.fill_between(x, f_one, f_three, color="blue", alpha=0.33)
plt.fill_between(x, f_three, f_two, color="green", alpha=0.33)
plt.xlabel(r"$p$")
plt.ylabel(r"$y(p)$")
plt.legend(loc="center")
plt.savefig("../assets/bounds.pdf")


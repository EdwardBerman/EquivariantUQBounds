import glob
import re
from collections import defaultdict
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
    plt.rcParams['text.usetex'] = False
    plt.rcParams['text.latex.preamble'] = r'\usepackage{amssymb}'

    return

set_rc_params(fontsize=20)


files = glob.glob('../data/test/*.npy')
grouped_files = defaultdict(list)
pattern = re.compile(r'_(\d+)\.npy$')

aleatoric_uncertainty = []

for file in files:
    match = pattern.search(file)
    if match:
        number = match.group(1)
        grouped_files[number].append(file)

for number, file_list in sorted(grouped_files.items()):
    labels_file = next(f for f in file_list if "labels" in f)
    mean_pred_file = next(f for f in file_list if "mean_pred" in f)
    al_uq_file = next(f for f in file_list if "al_std_dev" in f)
    ep_uq_file = next(f for f in file_list if "ep_std_dev" in f)
    mol_name_file = next(f for f in file_list if "mol_smile_name" in f)

    labels = np.load(labels_file)
    mean_pred = np.load(mean_pred_file)
    al_uq = np.load(al_uq_file)

    aleatoric_uncertainty.append(np.linalg.norm(al_uq))

    ep_uq = np.load(ep_uq_file)
    mol_name_data = np.load(mol_name_file, allow_pickle=True)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    x_axis = np.linspace(0, 1, len(mean_pred))

    ax[0].plot(x_axis, mean_pred, 'o', color='black', label='Predictions',linewidth=1, markersize=3)
    ax[0].plot(x_axis, labels, 'o', color='red', label='Labels',linewidth=1, markersize=3)
    ax[0].set_xlabel('Normalized Wavenumber')
    ax[0].set_ylabel(r'$\log_{10}$ Intensity')
    ax[0].set_title(f'{mol_name_data}')
    ax[0].legend(fontsize=8)

    ax[1].plot(x_axis, al_uq, 'o', color='blue', label='Aleatoric Uncertainty',linewidth=2, markersize=3)
    ax[1].plot(x_axis, ep_uq, 'o', color='green', label='Epistemic Uncertainty',linewidth=2, markersize=3)
    ax[1].set_xlabel('Normalized Wavenumber')
    ax[1].set_ylabel('Uncertainty')
    ax[1].set_title(f'{mol_name_data}')
    ax[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(f'../assets/spectra/{number}.pdf')

print(f'Mean aleatoric uncertainty: {np.mean(aleatoric_uncertainty)}')

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

set_rc_params(fontsize=20)

baseline_aleatoric_uncertainty = np.load('../data/aleatoric_uqs_gnn.npy')
equivariant_aleatoric_uncertainty = np.load('../data/aleatoric_uqs_egnn.npy')

baseline_epistemic_uncertainty = np.load('../data/epistemic_uqs_gnn.npy')
equivariant_epistemic_uncertainty = np.load('../data/epistemic_uqs_egnn.npy')

baseline_targets = np.load('../data/targets_gnn.npy')
equivariant_targets = np.load('../data/targets_egnn.npy')

baseline_predictions = np.load('../data/preds_gnn.npy')
equivariant_predictions = np.load('../data/preds_egnn.npy')

baseline_aleatoric_uncertainty = np.squeeze(baseline_aleatoric_uncertainty)
equivariant_aleatoric_uncertainty = np.squeeze(equivariant_aleatoric_uncertainty)
baseline_epistemic_uncertainty = np.squeeze(baseline_epistemic_uncertainty)
equivariant_epistemic_uncertainty = np.squeeze(equivariant_epistemic_uncertainty)
baseline_targets = np.squeeze(baseline_targets)
equivariant_targets = np.squeeze(equivariant_targets)
baseline_predictions = np.squeeze(baseline_predictions)
equivariant_predictions = np.squeeze(equivariant_predictions)

fig, ax = plt.subplots(2, 2, figsize=(8, 8))

sorted_indices = np.argsort(baseline_predictions)
sorted_baseline_predictions = baseline_predictions[sorted_indices]
sorted_baseline_targets = baseline_targets[sorted_indices]

sorted_indices = np.argsort(equivariant_predictions)
sorted_equivariant_predictions = equivariant_predictions[sorted_indices]
sorted_equivariant_targets = equivariant_targets[sorted_indices]

r2_baseline =   1 - np.sum((baseline_targets - baseline_predictions)**2) / np.sum((baseline_targets - np.mean(baseline_targets))**2)
r2_equivariant = 1 - np.sum((equivariant_targets - equivariant_predictions)**2) / np.sum((equivariant_targets - np.mean(equivariant_targets))**2)

ax[0,0].scatter(sorted_baseline_targets, sorted_baseline_predictions, color='blue', label=r'Baseline GNN, $R^2$={:.2f}'.format(r2_baseline), alpha=0.33)
ax[0,0].scatter(sorted_equivariant_targets, sorted_baseline_predictions, color='red', label=r'Equivariant GNN, $R^2$={:.2f}'.format(r2_equivariant), alpha=0.33)
ax[0,0].plot(sorted_baseline_targets, sorted_baseline_targets, color='black', linestyle='--', label='Perfect Prediction', alpha=0.33)
ax[0,0].legend(fontsize=8)
ax[0,0].set_xlabel(r'$\mu$ Predict (D)')
ax[0,0].set_ylabel(r'$\mu$ Label (D)')

ax[0,1].plot(baseline_aleatoric_uncertainty - equivariant_aleatoric_uncertainty, color='blue', label=r'$\Delta \sigma^2_{\rm Aleatoric}$', alpha=0.33)
ax[0,1].plot(baseline_epistemic_uncertainty - equivariant_epistemic_uncertainty, color='red', label=r'$\Delta \sigma^2_{\rm Epistemic}$', alpha=0.33)
ax[0,1].plot(np.zeros_like(baseline_aleatoric_uncertainty), color='black', linestyle='--', label='Zero Line')
ax[0,1].legend(fontsize=8)
ax[0,1].set_xlabel('Label')
ax[0,1].set_ylabel(r'$\Delta \sigma^2$')

ax[1,0].hist(baseline_epistemic_uncertainty, bins=50, color='blue', alpha=0.5, label='Baseline GNN Epistemic Uncertainty')
mean = np.mean(baseline_epistemic_uncertainty)
median = np.median(baseline_epistemic_uncertainty)
ax[1,0].hist(equivariant_epistemic_uncertainty, bins=50, color='red', alpha=0.5, label='Equivariant GNN Epistemic Uncertainty')
mean = np.mean(equivariant_epistemic_uncertainty)
median = np.median(equivariant_epistemic_uncertainty)
ax[1,0].set_xlabel(r'$\sigma^2_{\rm Epistemic}$')
ax[1,0].set_ylabel("Frequency")
ax[1,0].set_xlim([0.0, 2.5])
ax[1,0].legend(fontsize=8)

ax[1,1].hist(baseline_aleatoric_uncertainty, bins=50, color='blue', alpha=0.5, label='Baseline GNN Aleatoric Uncertainty')
mean = np.mean(baseline_aleatoric_uncertainty)
median = np.median(baseline_aleatoric_uncertainty)
ax[1,1].axvline(mean, color='blue', linestyle='--', label=f'Bleed {mean:.2f}')
ax[1,1].hist(equivariant_aleatoric_uncertainty, bins=50, color='red', alpha=0.5, label='Equivariant GNN Aleatoric Uncertainty')
mean = np.mean(equivariant_aleatoric_uncertainty)
median = np.median(equivariant_aleatoric_uncertainty)
ax[1,1].axvline(mean, color='red', linestyle='--', label=f'Bleed {mean:.2f}')
ax[1,1].set_xlabel(r'$\sigma^2_{\rm Aleatoric}$')
ax[1,1].set_ylabel("Frequency")
ax[1,1].set_xlim([0.0, 1.0])
ax[1,1].legend(fontsize=8)

plt.tight_layout()

plt.savefig('../assets/uq_bleed_dipole.pdf')


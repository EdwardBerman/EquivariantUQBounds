import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from matplotlib import rcParams, rc
from matplotlib.gridspec import GridSpec
import glob 
import re

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

file_list = glob.glob('../data/npy_files_biomarker/eval_uq/*.npy')

grouped_files = {}

for f in file_list:
    m = re.search(r'_(\d+)\.npy$', f)
    if m:
        group_key = m.group(1)
    else:
        group_key = 'none'
    grouped_files.setdefault(group_key, []).append(f)

for group, files in grouped_files.items():
    data = {}

    for f in files:
        if 'egnn' in f:
            if "aleatoric" in f:
                data['equivariant_aleatoric'] = np.load(f, allow_pickle=True)
            elif "epistemic" in f:
                data['equivariant_epistemic'] = np.load(f, allow_pickle=True)
            elif "target" in f:
                data['equivariant_target'] = np.load(f, allow_pickle=True)
            elif "preds" in f:
                data['equivariant_preds'] = np.load(f, allow_pickle=True)
        else:
            if "aleatoric" in f:
                data['baseline_aleatoric'] = np.load(f, allow_pickle=True)
            elif "epistemic" in f:
                data['baseline_epistemic'] = np.load(f, allow_pickle=True)
            elif "target" in f:
                data['baseline_target'] = np.load(f, allow_pickle=True)
            elif "preds" in f:
                data['baseline_preds'] = np.load(f, allow_pickle=True)

    equivariant_epistemic = data['equivariant_epistemic']
    equivariant_aleatoric = data['equivariant_aleatoric']
    equivariant_targets = data['equivariant_target']
    equivariant_predictions = data['equivariant_preds']

    baseline_epistemic = data['baseline_epistemic']
    baseline_aleatoric = data['baseline_aleatoric']
    baseline_targets = data['baseline_target']
    baseline_predictions = data['baseline_preds']

    sorted_indices = np.argsort(baseline_predictions)
    sorted_baseline_predictions = baseline_predictions[sorted_indices]
    sorted_baseline_targets = baseline_targets[sorted_indices]

    sorted_indices = np.argsort(equivariant_predictions)
    sorted_equivariant_predictions = equivariant_predictions[sorted_indices]
    sorted_equivariant_targets = equivariant_targets[sorted_indices]

    r2_baseline =   1 - np.sum((baseline_targets - baseline_predictions)**2) / np.sum((baseline_targets - np.mean(baseline_targets))**2)
    r2_equivariant = 1 - np.sum((equivariant_targets - equivariant_predictions)**2) / np.sum((equivariant_targets - np.mean(equivariant_targets))**2)

    baseline_mae = np.mean(np.abs(baseline_targets - baseline_predictions))
    equivariant_mae = np.mean(np.abs(equivariant_targets - equivariant_predictions))

    if np.abs(baseline_mae - equivariant_mae) < 0.25:
        group_number = group
        print(group_number)
        print(f"MAE baseline: {baseline_mae}")
        print(f"MAE equivariant: {equivariant_mae}")
        print(f"Aleatoric UQ baseline: {np.mean(baseline_aleatoric**2)}")
        print(f"Aleatoric UQ equivariant: {np.mean(equivariant_aleatoric**2)}")
        print("=" * 80)


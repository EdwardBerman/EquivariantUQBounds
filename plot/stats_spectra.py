import glob
import re
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from matplotlib import rcParams, rc
from matplotlib.gridspec import GridSpec
import tqdm

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

set_rc_params(fontsize=28)

files = glob.glob('../data/test/*.npy')
grouped_files = defaultdict(list)
pattern = re.compile(r'_(\d+)\.npy$')

for file in files:
    match = pattern.search(file)
    if match:
        number = match.group(1)
        grouped_files[number].append(file)

labels_np = np.zeros((32, 3501))
mean_pred_np = np.zeros((32, 3501))
al_uq_np = np.zeros((32, 3501))
ep_uq_np = np.zeros((32, 3501))

count = 0
for number, file_list in sorted(grouped_files.items()):
    labels_file = next(f for f in file_list if "labels" in f)
    mean_pred_file = next(f for f in file_list if "mean_pred" in f)
    al_uq_file = next(f for f in file_list if "al_std_dev" in f)
    ep_uq_file = next(f for f in file_list if "ep_std_dev" in f)
    mol_name_file = next(f for f in file_list if "mol_smile_name" in f)

    labels = np.load(labels_file)
    mean_pred = np.load(mean_pred_file)
    al_uq = np.load(al_uq_file)
    ep_uq = np.load(ep_uq_file)

    labels_np[count] = labels
    mean_pred_np[count] = mean_pred
    al_uq_np[count] = al_uq
    ep_uq_np[count] = np.sqrt(ep_uq)
    count += 1

print("Max label: ", np.max(labels_np))
print("Max mean_pred: ", np.max(mean_pred_np))
print("Min label: ", np.min(labels_np))
print("Min mean_pred: ", np.min(mean_pred_np))


def ENCE(y_true, y_pred, y_pred_std, bins=10):
    max_stds = np.max(y_pred_std, axis=0)
    min_stds = np.min(y_pred_std, axis=0)
    bins_edges = np.linspace(min_stds, max_stds, bins+1)

    ENCE = 0.0
    number_vecs = []

    for i in range(bins_edges.shape[1] - 1):
        for j in range(bins_edges.shape[0]):
            mask = (y_pred_std[:, j] >= bins_edges[j, i]) & (y_pred_std[:, j] < bins_edges[j, i+1])
            number_vectors = np.sum(mask)
            if number_vectors == 0:
                continue
            number_vecs.append(number_vectors)
            y_true_bin = y_true[mask]
            y_pred_bin = y_pred[mask]
            y_pred_std_bin = y_pred_std[mask]

            ENCE += np.mean(np.linalg.norm(y_pred_std_bin - np.abs(y_true_bin - y_pred_bin), axis=1))**2/ np.mean(np.linalg.norm(y_pred_std_bin, axis=1))**2 * number_vectors
    
    return ENCE / y_true.shape[0] , np.mean(number_vecs)

ENCE_bins = []
avg_bin_counts = []

for bins in tqdm.tqdm(range(1, 100, 1)):
    ENCE_in_bins, avg_bin_count = ENCE(labels_np, mean_pred_np, ep_uq_np, bins=bins)
    ENCE_bins.append(ENCE_in_bins)
    avg_bin_counts.append(avg_bin_count)

fig = plt.figure(figsize=(24, 12))
plt.plot(range(1, 100, 1), ENCE_bins, label="ENCE")
plt.xlabel("Number of bins")
plt.ylabel("ENCE")
plt.savefig("../assets/ENCE.pdf")
plt.close()

fig = plt.figure(figsize=(24, 12))
plt.plot(range(1, 100, 1), avg_bin_counts)
plt.xlabel("Number of bins")
plt.ylabel("Average Vector Per Bin")
plt.savefig("../assets/avg_bin_counts.pdf")



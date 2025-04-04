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

#files = glob.glob('../data/test/*.npy') 
files = glob.glob('../data/final_qm9_rotated/*.npy') # final_qm9_rotated
grouped_files = defaultdict(list)
pattern = re.compile(r'_(\d+)\.npy$')

for file in files:
    match = pattern.search(file)
    if match:
        number = match.group(1)
        grouped_files[number].append(file)

labels_np = np.zeros((128, 3501))
mean_pred_np = np.zeros((128, 3501))
al_uq_np = np.zeros((128, 3501))
ep_uq_np = np.zeros((128, 3501))

labels_augmented_np = np.zeros((128, 3501))

count = 0
for number, file_list in sorted(grouped_files.items()):
    labels_file = next(f for f in file_list if "labels" in f)
    mean_pred_file = next(f for f in file_list if "mean_pred" in f)
    al_uq_file = next(f for f in file_list if "al_std_dev" in f)
    ep_uq_file = next(f for f in file_list if "ep_std_dev" in f)
    mol_name_file = next(f for f in file_list if "mol_smile_name" in f)

    if "0.npy" in labels_file:
        labels_augmented = np.load(labels_file)
        labels_augmented = -labels_augmented
        labels_augmented_np[count] = labels_augmented
    if "2.npy" in labels_file:
        labels_augmented = np.load(labels_file)
        labels_augmented = -labels_augmented
        labels_augmented_np[count] = labels_augmented
    else:
        labels_augmented = np.load(labels_file)
        labels_augmented_np[count] = labels_augmented
        
    labels = np.load(labels_file)
    labels_np[count] = labels
    mean_pred = np.load(mean_pred_file)
    al_uq = np.load(al_uq_file)
    ep_uq = np.load(ep_uq_file)
    mean_pred_np[count] = mean_pred
    al_uq_np[count] = np.sqrt(al_uq)
    ep_uq_np[count] = np.sqrt(ep_uq)
    count += 1

print("Max label: ", np.max(labels_np))
print("Max mean_pred: ", np.max(mean_pred_np))
print("Min label: ", np.min(labels_np))
print("Min mean_pred: ", np.min(mean_pred_np))
print("Median label: ", np.median(labels_np))

print(np.min(np.linalg.norm(np.sqrt(2/np.pi)*ep_uq_np, axis=1)**2))
print(np.max(np.linalg.norm(np.sqrt(2/np.pi)*ep_uq_np, axis=1)**2))


def ENCE(y_true, y_pred, y_pred_std, bins=10, max_label=3.3, min_label=-3.3):
    max_stds = np.max(y_pred_std, axis=0)
    min_stds = np.min(y_pred_std, axis=0)
    bins_edges = np.linspace(min_stds, max_stds, bins+1)

    ENCE = 0.0
    upper_bound = 1.0
    max_error = 5 * np.ones_like(y_true)
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
            diff_max = np.abs(max_label - y_true_bin)
            diff_min = np.abs(min_label - y_true_bin)
            max_error_bin = np.where(diff_max < diff_min, diff_min, diff_max)

            ENCE += number_vectors * np.mean(np.linalg.norm(np.sqrt(2/np.pi)*y_pred_std_bin - np.abs(y_true_bin - y_pred_bin), axis=1)**2)/ np.mean(np.linalg.norm(np.sqrt(2/np.pi)*y_pred_std_bin, axis=1)**2) 
            upper_bound += number_vectors * np.mean(np.linalg.norm(max_error_bin, axis=1)**2)/ 6.945443911561127 #* number_vectors
    
    return ENCE / y_true.shape[0] , np.mean(number_vecs), upper_bound / y_true.shape[0]

ENCE_bins_correct_augmentation = []
ENCE_bins_incorrect_augmentation = []
avg_bin_counts = []
upper_bound_correct = []
upper_bound_incorrect = []

for bins in tqdm.tqdm(range(1, 100, 1)):
    ENCE_in_bins, avg_bin_count, ub_c = ENCE(labels_np, mean_pred_np, ep_uq_np, bins=bins)
    ENCE_bins_correct_augmentation.append(ENCE_in_bins)

    ENCE_in_bins, avg_bin_count, ub_i = ENCE(labels_augmented_np, mean_pred_np, ep_uq_np, bins=bins)
    ENCE_bins_incorrect_augmentation.append(ENCE_in_bins)

    avg_bin_counts.append(avg_bin_count)
    upper_bound_correct.append(ub_c)
    upper_bound_incorrect.append(ub_i)

fig = plt.figure(figsize=(24, 12))
plt.plot(range(1, 100, 1), ENCE_bins_correct_augmentation, label="ENCE")
plt.xlabel("Number of bins")
plt.ylabel("ENCE")
#plt.yscale("log")
plt.savefig("../assets/ENCE.pdf")
plt.close()

fig = plt.figure(figsize=(24, 12))
plt.plot(range(1, 100, 1), avg_bin_counts)
plt.xlabel("Number of bins")
plt.ylabel("Average Vector Per Bin")
plt.yscale("log")
plt.savefig("../assets/avg_bin_counts.pdf")
plt.close()

fig = plt.figure(figsize=(24, 12))
plt.plot(range(1, 100, 1), ENCE_bins_correct_augmentation, label="Computed ENCE (Correct Augmentation)")
plt.plot(range(1, 100, 1), ENCE_bins_incorrect_augmentation, label="Computed ENCE (Incorrect Augmentation)")
plt.plot(range(1, 100, 1), upper_bound_correct, label="Upper Bound")
#plt.plot(range(1, 100, 1), upper_bound_incorrect, label="Upper Bound (Incorrect Augmentation)")
plt.xlabel("Number of bins")
plt.ylabel("ENCE")
plt.yscale("log")
plt.legend()
plt.savefig("../assets/upper_bound_vs_computed.pdf")
plt.close()

fig, ax = plt.subplots(1, 2, figsize=(48, 12))
ax[0].plot(range(1, 100, 1), ENCE_bins_correct_augmentation, label="Computed ENCE (Correct Augmentation)")
ax[0].plot(range(1, 100, 1), ENCE_bins_incorrect_augmentation, label="Computed ENCE (Incorrect Augmentation)")
ax[0].plot(range(1, 100, 1), upper_bound_correct, label="Upper Bound")
ax[0].set_xlabel("Number of bins")
ax[0].set_ylabel("ENCE")
ax[0].set_yscale("log")
ax[0].legend()

length_normalized_correct = np.array(ENCE_bins_correct_augmentation)
length_normalized_incorrect = np.array(ENCE_bins_incorrect_augmentation) 
length_normalized_upper_bound = np.array(upper_bound_correct) 

ax[1].plot(range(1, 100, 1), length_normalized_correct, label="Computed ENCE (Correct Augmentation)")
ax[1].plot(range(1, 100, 1), length_normalized_incorrect, label="Computed ENCE (Incorrect Augmentation)")
ax[1].plot(range(1, 100, 1), length_normalized_upper_bound, label="Upper Bound")
ax[1].set_xlabel("Number of bins")
ax[1].set_ylabel(r"ENCE")
ax[1].legend()

plt.savefig("../assets/multi_scale_upper_bound_vs_computed.pdf")


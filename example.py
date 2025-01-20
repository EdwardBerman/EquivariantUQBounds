import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from matplotlib import rcParams, rc
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
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

node_colors = ['lightblue', 'red']
node_labels = {1: r'$x_0$', 2: r'$x_1$'}

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

G  = nx.DiGraph()

G.add_edges_from([(1, 1), (2, 2), (1, 2), (2, 1)])

pos = {1: (-1, 0), 2: (1, 0)}

nx.draw(G, pos, with_labels=True, ax=axes[0,1], node_color=node_colors, edge_color='gray', node_size=2000, font_size=16,connectionstyle="arc3,rad=0.3", labels=node_labels)

axes[0,1].set_title("Orbit 1", fontsize=28)

for spine in axes[0,0].spines.values():
    spine.set_edgecolor('black')  # Set the border color
    spine.set_linewidth(2)       # Set the border thickness

labels = [r'$y = 0$', r'$y = 1$']
ylabels = [r'$0$', r'$p_1$', r'$1 - p_1$']
axes[0,2].set_xticks([0, 1])
axes[0,2].set_xticklabels(labels)
axes[0,2].set_yticks([0, 0.7, 0.3])
axes[0,2].set_yticklabels(ylabels)
axes[0,2].bar([0, 1], [0.7, 0.3], color='lightgreen', edgecolor='black', linewidth=2)
axes[0,2].set_title(r"$Y \times \hat{P}$ pairs", fontsize=28)
for spine in axes[0,1].spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(2)

# Adjust layout to avoid overlap

G  = nx.DiGraph()

G.add_edges_from([(1, 1), (2, 2), (1, 2), (2, 1)])

pos = {1: (-1, 0), 2: (1, 0)}

node_colors = ['lightblue', 'lightblue']

node_labels = {1: r'$x_2$', 2: r'$x_3$'}

nx.draw(G, pos, with_labels=True, ax=axes[1,1], node_color=node_colors, edge_color='gray', node_size=2000, font_size=16,connectionstyle="arc3,rad=0.3", labels=node_labels)

axes[1,1].set_title("Orbit 2", fontsize=28)

for spine in axes[1,0].spines.values():
    spine.set_edgecolor('black')  # Set the border color
    spine.set_linewidth(2)       # Set the border thickness

labels = [r'$y = 0$', r'$y = 1$']
ylabels = [r'$0$', r'$p_2$', r'$1 - p_2$']
axes[1,2].set_xticks([0, 1])
axes[1,2].set_xticklabels(labels)
axes[1,2].set_yticks([0, 0.4, 0.6])
axes[1,2].set_yticklabels(ylabels)
axes[1,2].bar([0, 1], [0.4, 0.6], color='lightgreen', edgecolor='black', linewidth=2)
#axes[1,2].set_title("Probability Density", fontsize=28)
for spine in axes[1,1].spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(2)

# pdf plot 
x = np.array([0, 1, 2, 3])
x_labels = [r'$x_0$', r'$x_1$', r'$x_2$', r'$x_3$']
y = np.array([0.3, 0.4, 0.1, 0.2])
axes[0,0].bar(x, y, color='lightgreen', edgecolor='black', linewidth=2)
axes[0,0].set_xticks(x)
axes[0,0].set_xticklabels(x_labels)
axes[0,0].set_title("Probability Density q(x)", fontsize=28)
axes[0,0].set_ylabel("q(x)")

def expected_value_h_p_upper_bound(p1, p2):
    p1 = np.maximum(p1, 1 - p1)
    p2 = np.maximum(p2, 1 - p2)

    orbit_one_sum = p1 * 0.3 + p1 * 0.4
    orbit_two_sum = p2 * 0.1 + p2 * 0.2

    fundamental_domain_sum = orbit_one_sum + orbit_two_sum
    return fundamental_domain_sum

heatmap = np.zeros((100, 100))
for i, p1 in enumerate(np.linspace(0, 1, 100)):
    for j, p2 in enumerate(np.linspace(0, 1, 100)):
        heatmap[i, j] = expected_value_h_p_upper_bound(p1, p2)

axes[1,0].imshow(heatmap, cmap='viridis', extent=[0, 1, 0, 1], origin='lower')
axes[1,0].set_xlabel(r'$p_1$')
axes[1,0].set_ylabel(r'$p_2$')
axes[1,0].set_title(r'$\mathbb{E}[h(p)]$' + ' upper bound', fontsize=28)

plt.tight_layout()

plt.savefig("assets/example.pdf")

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

fig, axes = plt.subplots(2, 2, figsize=(12, 12))

G  = nx.DiGraph()

G.add_edges_from([(1, 1), (2, 2), (1, 2), (2, 1)])

pos = {1: (-1, 0), 2: (1, 0)}

nx.draw(G, pos, with_labels=True, ax=axes[0,0], node_color=node_colors, edge_color='gray', node_size=2000, font_size=16,connectionstyle="arc3,rad=0.3")

axes[0,0].set_title("Orbit 1", fontsize=16)

for spine in axes[0,0].spines.values():
    spine.set_edgecolor('black')  # Set the border color
    spine.set_linewidth(2)       # Set the border thickness

labels = [r'$y = 0$', r'$y = 1$']
ylabels = [r'$0$', r'$p$', r'$1 - p$']
axes[0,1].set_xticks([0, 1])
axes[0,1].set_xticklabels(labels)
axes[0,1].set_yticks([0, 0.7, 0.3])
axes[0,1].set_yticklabels(ylabels)
axes[0,1].bar([0, 1], [0.7, 0.3], color='lightgreen', edgecolor='black', linewidth=2)
axes[0,1].set_title("Probability Density", fontsize=16)
for spine in axes[0,1].spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(2)

# Adjust layout to avoid overlap

G  = nx.DiGraph()

G.add_edges_from([(1, 1), (2, 2), (1, 2), (2, 1)])

pos = {1: (-1, 0), 2: (1, 0)}

node_colors = ['lightblue', 'lightblue']

nx.draw(G, pos, with_labels=True, ax=axes[1,0], node_color=node_colors, edge_color='gray', node_size=2000, font_size=16,connectionstyle="arc3,rad=0.3")

axes[1,0].set_title("Orbit 2", fontsize=16)

for spine in axes[1,0].spines.values():
    spine.set_edgecolor('black')  # Set the border color
    spine.set_linewidth(2)       # Set the border thickness

labels = [r'$y = 0$', r'$y = 1$']
ylabels = [r'$0$', r'$p$', r'$1 - p$']
axes[1,1].set_xticks([0, 1])
axes[1,1].set_xticklabels(labels)
axes[1,1].set_yticks([0, 0.4, 0.6])
axes[1,1].set_yticklabels(ylabels)
axes[1,1].bar([0, 1], [0.4, 0.6], color='lightgreen', edgecolor='black', linewidth=2)
axes[1,1].set_title("Probability Density", fontsize=16)
for spine in axes[1,1].spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(2)



plt.tight_layout()





plt.savefig("assets/example.pdf")

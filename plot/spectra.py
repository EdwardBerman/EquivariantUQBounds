import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.interpolate import interp1d
from matplotlib import rcParams, rc
from matplotlib.gridspec import GridSpec
import matplotlib.patches as patches
import networkx as nx

def set_np_seed(seed):
    np.random.seed(seed)
    return

set_np_seed(42)

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

set_rc_params(fontsize=40)

fig, ax = plt.subplots(3, 4, figsize=(42, 36))

G = nx.Graph()

positions = {
    (0, 0): (0, 0),
    (1, 0): (1, 0),
    (0, 1): (0, 1),
    (-1, 0): (-1, 0),
    (0, -1): (0, -1)
}

node_labels = {
    (0, 0): r'$C$',
    (1, 0): r'$H$',
    (0, 1): r'$H$',
    (-1, 0): r'$H$',
    (0, -1): r'$H$'
}

for node, pos in positions.items():
    G.add_node(node, pos=pos, label=node_labels[node])

edges = [
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1)),
    ((0, 0), (-1, 0)),
    ((0, 0), (0, -1))
]
G.add_edges_from(edges)

ax[0,0].grid(True, which='both', linestyle='--', linewidth=8)

node_sizes = [7200 if node == (0, 0) else 3600 for node in G.nodes()]
node_colors = ["lightblue" if node == (0, 0) else "lightgreen" for node in G.nodes()]

nx.draw(
    G, 
    pos=positions, 
    with_labels=True, 
    node_size=node_sizes,
    node_color=node_colors,
    font_weight="bold",
    font_size=24,
    ax=ax[0, 0]
)

ax[0, 0].set_title(r'$CH_4 (\times)$')

G = nx.Graph()

sqrt2_over_2 = round(np.sqrt(2) / 2, 2)
positions = {
    (0, 0): (0, 0),
    (sqrt2_over_2, sqrt2_over_2): (sqrt2_over_2, sqrt2_over_2),
    (-sqrt2_over_2, sqrt2_over_2): (-sqrt2_over_2, sqrt2_over_2),
    (-sqrt2_over_2, -sqrt2_over_2): (-sqrt2_over_2, -sqrt2_over_2),
    (sqrt2_over_2, -sqrt2_over_2): (sqrt2_over_2, -sqrt2_over_2)
}

node_labels = {
    (0, 0): r'$C$',
    (sqrt2_over_2, sqrt2_over_2): r'$H$',
    (-sqrt2_over_2, sqrt2_over_2): r'$H$',
    (-sqrt2_over_2, -sqrt2_over_2): r'$H$',
    (sqrt2_over_2, -sqrt2_over_2): r'$H$'
}

for node, pos in positions.items():
    G.add_node(node, pos=pos, label=node_labels[node])

edges = [
    ((0, 0), (sqrt2_over_2, sqrt2_over_2)),
    ((0, 0), (-sqrt2_over_2, sqrt2_over_2)),
    ((0, 0), (-sqrt2_over_2, -sqrt2_over_2)),
    ((0, 0), (sqrt2_over_2, -sqrt2_over_2))
]
G.add_edges_from(edges)

ax[0,1].grid(True, which='both', linestyle='--', linewidth=3)

node_sizes = [7200 if node == (0, 0) else 3600 for node in G.nodes()]
node_colors = ["lightblue" if node == (0, 0) else "lightgreen" for node in G.nodes()]

nx.draw(
    G, 
    pos=positions, 
    with_labels=True, 
    node_size=node_sizes,
    node_color=node_colors,
    font_weight="bold",
    font_size=24,
    ax=ax[0, 1]
)

G = nx.Graph()

positions = {
    (0, 0): (0, 0),
    (1, 0): (1, 0),
    (0, 1): (0, 1),
}

node_labels = {
    (0, 0): r'$O$',
    (1, 0): r'$H$',
    (0, 1): r'$H$',
}

for node, pos in positions.items():
    G.add_node(node, pos=pos, label=node_labels[node])

edges = [
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1)),
]
G.add_edges_from(edges)


node_sizes = [7200 if node == (0, 0) else 3600 for node in G.nodes()]
node_colors = ["lightblue" if node == (0, 0) else "lightgreen" for node in G.nodes()]

nx.draw(
    G, 
    pos=positions, 
    with_labels=True, 
    node_size=node_sizes,
    node_color=node_colors,
    font_weight="bold",
    font_size=24,
    ax=ax[1, 0]
)

G = nx.Graph()

positions = {
    (0, 0): (0, 0),
    (-1, 0): (-1, 0),
    (0, 1): (0, 1),
}

node_labels = {
    (0, 0): r'$S$',
    (-1, 0): r'$O$',
    (0, 1): r'$O$',
}

for node, pos in positions.items():
    G.add_node(node, pos=pos, label=node_labels[node])

edges = [
    ((0, 0), (-1, 0)),
    ((0, 0), (0, 1)),
]
G.add_edges_from(edges)


node_sizes = [7200 if node == (0, 0) else 3600 for node in G.nodes()]
node_colors = ["lightblue" if node == (0, 0) else "lightgreen" for node in G.nodes()]

nx.draw(
    G, 
    pos=positions, 
    with_labels=True, 
    node_size=node_sizes,
    node_color=node_colors,
    font_weight="bold",
    font_size=24,
    ax=ax[1, 1]
)

G = nx.Graph()

positions = {
    (0, 0): (0, 0),
    (-1, 0): (-1, 0),
    (0, 1): (0, 1),
    (1, 0): (1, 0),
}

node_labels = {
    (0, 0): r'$N$',
    (-1, 0): r'$H$',
    (0, 1): r'$H$',
    (1, 0): r'$H$',
}

for node, pos in positions.items():
    G.add_node(node, pos=pos, label=node_labels[node])

edges = [
    ((0, 0), (-1, 0)),
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
]
G.add_edges_from(edges)


node_sizes = [7200 if node == (0, 0) else 3600 for node in G.nodes()]
node_colors = ["lightblue" if node == (0, 0) else "lightgreen" for node in G.nodes()]

nx.draw(
    G, 
    pos=positions, 
    with_labels=True, 
    node_size=node_sizes,
    node_color=node_colors,
    font_weight="bold",
    font_size=24,
    ax=ax[2, 1]
)



ax[0, 0].set_title(r'$CH_4 (\times)$')
ax[0, 1].set_title(r'$CH_4 (+)$')
ax[1, 0].set_title(r'$H_20$')
ax[1, 1].set_title(r'$SO_2$')
ax[2, 1].set_title(r'$NH_3$')

fig.tight_layout()
subplots_to_include = [(0, 0), (0, 1), (1, 0), (1, 1)]
xmin = min(ax[i, j].get_position().xmin for i, j in subplots_to_include)
xmax = max(ax[i, j].get_position().xmax for i, j in subplots_to_include)
ymin = min(ax[i, j].get_position().ymin for i, j in subplots_to_include)
ymax = max(ax[i, j].get_position().ymax for i, j in subplots_to_include)

border_color = "red"
border_width = 9
rect = patches.Rectangle(
    (xmin, ymin),  # Bottom-left corner
    xmax - xmin,   # Width of the rectangle
    ymax - ymin,   # Height of the rectangle
    linewidth=border_width,
    edgecolor=border_color,
    facecolor="none",  # Transparent fill
    transform=fig.transFigure,  # Use figure coordinates
    zorder=10  # Above other elements
)
#fig.patches.append(rect)

top_subplots_to_include = [(0, 0), (0, 1)]
xmin_small = min(ax[i, j].get_position().xmin for i, j in top_subplots_to_include)
xmax_small = max(ax[i, j].get_position().xmax for i, j in top_subplots_to_include)
ymin_small = min(ax[i, j].get_position().ymin for i, j in top_subplots_to_include)

# Extend ymax above the axis title (adding padding)
ymax_small = max(ax[i, j].get_position().ymax for i, j in top_subplots_to_include) + 0.02

# Add a smaller black rectangle
border_color_small = "black"
border_width_small = 5
rect_small = patches.Rectangle(
    (xmin_small, ymin_small),  # Bottom-left corner
    xmax_small - xmin_small,  # Width of the rectangle
    ymax_small - ymin_small,  # Height of the rectangle
    linewidth=border_width_small,
    edgecolor=border_color_small,
    facecolor="none",  # Transparent fill
    transform=fig.transFigure,  # Use figure coordinates
    zorder=11  # Above the red rectangle
)
#fig.patches.append(rect_small)

plt.subplots_adjust(top=0.92, bottom=0.08, left=0.08, right=0.92, hspace=0.3, wspace=0.3)

methane_spectra = np.load('../data/methane_spectra.npy')
water_spectra = np.load('../data/water_spectra.npy')
sulfur_dioxide_spectra = np.load('../data/sulfur_dioxide.npy')
ammonia_spectra = np.load('../data/amonia_spectra.npy')

sulfur_dioxide_spectra_x = sulfur_dioxide_spectra[:, 0]
sulfur_dioxide_spectra_y = sulfur_dioxide_spectra[:, 1]

methane_spectra = methane_spectra / np.max(methane_spectra)
water_spectra = water_spectra / np.max(water_spectra)
ammonia_spectra = ammonia_spectra / np.max(ammonia_spectra)

ax[0, 2].plot(methane_spectra, color='black', label=r'f(CH$_4(+))$')
ax[0, 2].plot(methane_spectra, color='red', label=r'$f(CH_4(\times))$')
ax[0, 2].plot(methane_spectra, color='blue', label=r'$h(Orbit)$')
ax[0, 2].legend()
ax[0, 2].set_title(r'$CH_4(+/\times)$ Transition Spectra')

ax[1, 2].plot(water_spectra, color='black', label=r'f(H$_2O)$')
ax[1, 2].plot(sulfur_dioxide_spectra_x, sulfur_dioxide_spectra_y, color='red', label=r'$f(SO_2)$')

interp_function = interp1d(sulfur_dioxide_spectra_x, sulfur_dioxide_spectra_y, kind='linear', fill_value="extrapolate")
water_x = np.linspace(500, 4000, len(water_spectra))
sulfur_dioxide_y_interpolated = interp_function(water_x)
average_spectra = (water_spectra + sulfur_dioxide_y_interpolated) / 2
ax[1, 2].plot(average_spectra, color='blue', label=r'$h(Orbit)$')
ax[1, 2].legend()
ax[1, 2].set_title(r'$H_2O/SO_2$ Transition Spectra')

ax[2, 2].plot(ammonia_spectra, color='black', label=r'f(NH$_3)$')
ax[2, 2].plot(ammonia_spectra, color='red', label=r'$g(Orbit)$')
ax[2, 2].legend()
ax[2, 2].set_title(r'$NH_3$ Transition Spectra')

x = np.array([0, 1, 2, 3, 4])
x_labels = [r'$CH_4(+)$', r'$CH_4(\times)$', r'$H_20$', r'$SO_2$', r'$NH_3$']
y = np.array([0.125, 0.125, 0.125, 0.125, 0.5])
ax[2,0].bar(x, y, color='lightgreen', edgecolor='black', linewidth=2)
ax[2,0].set_xticks(x)
ax[2,0].set_xticklabels(x_labels, fontsize=24)
ax[2,0].set_title("Probability Density")
ax[2,0].set_xlabel("Molecule")
ax[2,0].set_ylabel("p(x)")


x = np.linspace(0, 4000, 4000)
noise = np.random.uniform(-0.1, 0.1, 4000)
y = np.sin(2 * np.pi * (1/500)*x) + 1.2 + noise
ax[0, 3].plot(x, y, color='black', label=r'$\vec{\sigma}_1$')
ax[0, 3].set_yticks([])
ax[0, 3].set_title(r'$\vec{\sigma^2}_1$')

ax[1, 3].plot(x, y, color='black', label=r'$\vec{\sigma}_1$')
ax[1, 3].set_yticks([])
ax[1, 3].set_title(r'$\vec{\sigma^2}_1$')

shift = 500
y_new = (np.sin(2 * np.pi * (1/500)*x + shift) + 1.2 + noise) - np.sqrt(2)

ax[2, 3].plot(x, y_new, color='black', label=r'$\vec{\sigma}_2$')
ax[2, 3].set_yticks([])
ax[2, 3].set_title(r'$\vec{\sigma^2}_2$')

print((np.linalg.norm(water_spectra - average_spectra) + np.linalg.norm(water_spectra - sulfur_dioxide_y_interpolated)) / 2)

plt.savefig('../assets/spectra.pdf', dpi=300)

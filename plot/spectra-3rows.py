"""
file: spectra-3rows.py
purpose: visualizes the data in methane_spectra.npy, water_spectra.npy, 
         sulfur_dioxide.npy, amonia_spectra.npy into three rows of plots
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.interpolate import interp1d
from matplotlib import rcParams, rc
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
        fontsize = 16
    else:
        fontsize = int(fontsize)

    rc('font',**{'family':'serif'})
    rc('text', usetex=True)

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

set_rc_params(fontsize=32)

# --------------------------------------------------------------------
# load data (do this once, reuse in each figure)
# --------------------------------------------------------------------
methane_spectra = np.load('../data/methane_spectra.npy')
water_spectra   = np.load('../data/water_spectra.npy')
so2_data        = np.load('../data/sulfur_dioxide.npy')
ammonia_spectra = np.load('../data/amonia_spectra.npy')

so2_x = so2_data[:, 0]
so2_y = so2_data[:, 1]

# Normalize
methane_spectra = methane_spectra / np.max(methane_spectra)
water_spectra   = water_spectra   / np.max(water_spectra)
ammonia_spectra = ammonia_spectra / np.max(ammonia_spectra)

# For combined H2O + SO2
interp_function = interp1d(so2_x, so2_y, kind='linear', fill_value="extrapolate")
water_x = np.linspace(500, 4000, len(water_spectra))
so2_yi  = interp_function(water_x)
avg_spectra = (water_spectra + so2_yi) / 2

# --------------------------------------------------------------------
# first figure: row 1 (CH4 structures and CH4 transition spectra)
# --------------------------------------------------------------------
fig1, ax1 = plt.subplots(1, 4, figsize=(42, 12))

# ---- Subplot (1,0): CH4 (x) ----
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
edges = [((0, 0),(1, 0)),
         ((0, 0),(0, 1)),
         ((0, 0),(-1, 0)),
         ((0, 0),(0, -1))]
G.add_edges_from(edges)
node_sizes = [7200 if node==(0,0) else 3600 for node in G.nodes()]
node_colors= ["lightblue" if node==(0,0) else "lightgreen" for node in G.nodes()]
nx.draw(G,
        pos=positions,
        with_labels=True,
        node_size=node_sizes,
        node_color=node_colors,
        font_weight="bold",
        font_size=24,
        ax=ax1[0])
ax1[0].grid(True, which='both', linestyle='--', linewidth=8)
ax1[0].set_title(r'$CH_4 (\times)$')

# ---- Subplot (1,1): CH4 (+) ----
G2 = nx.Graph()
s2 = round(np.sqrt(2)/2, 2)
positions2 = {
    (0, 0): (0, 0),
    (s2,  s2): (s2,  s2),
    (-s2, s2): (-s2, s2),
    (-s2,-s2): (-s2,-s2),
    ( s2,-s2): ( s2,-s2)
}
node_labels2 = {
    (0, 0): r'$C$',
    (s2,  s2): r'$H$',
    (-s2, s2): r'$H$',
    (-s2,-s2): r'$H$',
    ( s2,-s2): r'$H$'
}
for node, pos in positions2.items():
    G2.add_node(node, pos=pos, label=node_labels2[node])
edges2 = [((0,0),( s2, s2)),
          ((0,0),(-s2, s2)),
          ((0,0),(-s2,-s2)),
          ((0,0),( s2,-s2))]
G2.add_edges_from(edges2)
node_sizes2 = [7200 if node==(0,0) else 3600 for node in G2.nodes()]
node_colors2= ["lightblue" if node==(0,0) else "lightgreen" for node in G2.nodes()]
nx.draw(G2,
        pos=positions2,
        with_labels=True,
        node_size=node_sizes2,
        node_color=node_colors2,
        font_weight="bold",
        font_size=24,
        ax=ax1[1])
ax1[1].grid(True, which='both', linestyle='--', linewidth=3)
ax1[1].set_title(r'$CH_4 (+)$')

# ---- Subplot (1,2): CH4 transition spectra ----
ax1[2].plot(methane_spectra, color='black', label=r'$f(CH_4(+))$')
ax1[2].plot(methane_spectra, color='red',   label=r'$f(CH_4(\times))$')
ax1[2].plot(methane_spectra, color='blue',  label=r'$h(Orbit)$')
ax1[2].legend()
ax1[2].set_title(r'$CH_4(+/\times)$ Transition Spectra')

# ---- Subplot (1,3): sigma plot 1 ----
x = np.linspace(0,4000,4000)
noise = np.random.uniform(-0.1, 0.1, 4000)
y = np.sin(2*np.pi*(1/500)*x) + 1.2 + noise
ax1[3].plot(x, y, color='black')
ax1[3].set_yticks([])
ax1[3].set_title(r'$\vec{\sigma^2}_1$')

fig1.tight_layout()
plt.savefig('../assets/spectra_row1.pdf', dpi=300)
plt.close(fig1)

# --------------------------------------------------------------------
# Second figure: row 2 (H2O, SO2, their spectra, and second sigma plot)
# --------------------------------------------------------------------
fig2, ax2 = plt.subplots(1, 4, figsize=(42, 12))

# ---- Subplot (2,0): H2O structure ----
G3 = nx.Graph()
positions3 = {(0,0):(0,0), (1,0):(1,0), (0,1):(0,1)}
node_labels3= {(0,0):r'$O$', (1,0):r'$H$', (0,1):r'$H$'}
for node, pos in positions3.items():
    G3.add_node(node, pos=pos, label=node_labels3[node])
edges3 = [((0,0),(1,0)), ((0,0),(0,1))]
G3.add_edges_from(edges3)
node_sizes3 = [7200 if node==(0,0) else 3600 for node in G3.nodes()]
node_colors3= ["lightblue" if node==(0,0) else "lightgreen" for node in G3.nodes()]
nx.draw(G3,
        pos=positions3,
        with_labels=True,
        node_size=node_sizes3,
        node_color=node_colors3,
        font_weight="bold",
        font_size=24,
        ax=ax2[0])
ax2[0].set_title(r'$H_2O$')

# ---- subplot (2,1): SO2 structure ----
G4 = nx.Graph()
positions4 = {(0,0):(0,0), (-1,0):(-1,0), (0,1):(0,1)}
node_labels4= {(0,0):r'$S$', (-1,0):r'$O$', (0,1):r'$O$'}
for node, pos in positions4.items():
    G4.add_node(node, pos=pos, label=node_labels4[node])
edges4 = [((0,0),(-1,0)), ((0,0),(0,1))]
G4.add_edges_from(edges4)
node_sizes4 = [7200 if node==(0,0) else 3600 for node in G4.nodes()]
node_colors4= ["lightblue" if node==(0,0) else "lightgreen" for node in G4.nodes()]
nx.draw(G4,
        pos=positions4,
        with_labels=True,
        node_size=node_sizes4,
        node_color=node_colors4,
        font_weight="bold",
        font_size=24,
        ax=ax2[1])
ax2[1].set_title(r'$SO_2$')

# ---- subplot (2,2): H2O / SO2 transition spectra ----
ax2[2].plot(water_spectra, color='black', label=r'$f(H_2O)$')
ax2[2].plot(so2_x, so2_y,  color='red',   label=r'$f(SO_2)$')
ax2[2].plot(avg_spectra,   color='blue',  label=r'$h(Orbit)$')
ax2[2].legend()
ax2[2].set_title(r'$H_2O/SO_2$ Transition Spectra')

# ---- subplot (2,3): sigma plot 2 ----
ax2[3].plot(x, y, color='black')
ax2[3].set_yticks([])
ax2[3].set_title(r'$\vec{\sigma^2}_1$')

fig2.tight_layout()
plt.savefig('../assets/spectra_row2.pdf', dpi=300)
plt.close(fig2)

# --------------------------------------------------------------------
# third figure: row 3 (Probability bar, NH3, NH3 spectra, and sigma shift)
# --------------------------------------------------------------------
fig3, ax3 = plt.subplots(1, 4, figsize=(42, 12))

# ---- subplot (3,0): Probability density bar ----
xvals = np.array([0, 1, 2, 3, 4])
x_labels = [r'$CH_4(+)$', r'$CH_4(\times)$', r'$H_2O$', r'$SO_2$', r'$NH_3$']
yvals = np.array([0.125, 0.125, 0.125, 0.125, 0.5])
ax3[0].bar(xvals, yvals, color='lightgreen', edgecolor='black', linewidth=2)
ax3[0].set_xticks(xvals)
ax3[0].set_xticklabels(x_labels, fontsize=24)
ax3[0].set_title("Probability Density")
ax3[0].set_xlabel("Molecule")
ax3[0].set_ylabel("p(x)")

# ---- subplot (3,1): NH3 structure ----
G5 = nx.Graph()
positions5 = {(0,0):(0,0), (-1,0):(-1,0), (0,1):(0,1), (1,0):(1,0)}
node_labels5= {(0,0):r'$N$', (-1,0):r'$H$', (0,1):r'$H$', (1,0):r'$H$'}
for node, pos in positions5.items():
    G5.add_node(node, pos=pos, label=node_labels5[node])
edges5 = [((0,0),(-1,0)), ((0,0),(0,1)), ((0,0),(1,0))]
G5.add_edges_from(edges5)
node_sizes5 = [7200 if node==(0,0) else 3600 for node in G5.nodes()]
node_colors5= ["lightblue" if node==(0,0) else "lightgreen" for node in G5.nodes()]
nx.draw(G5,
        pos=positions5,
        with_labels=True,
        node_size=node_sizes5,
        node_color=node_colors5,
        font_weight="bold",
        font_size=24,
        ax=ax3[1])
ax3[1].set_title(r'$NH_3$')

# ---- subplot (3,2): NH3 transition spectra ----
ax3[2].plot(ammonia_spectra, color='black', label=r'$f(NH_3)$')
ax3[2].plot(ammonia_spectra, color='red',   label=r'$h(Orbit)$')
ax3[2].legend()
ax3[2].set_title(r'$NH_3$ Transition Spectra')

# ---- subplot (3,3): sigma plot 3 (shifted) ----
shift = 500
y_new = (np.sin(2*np.pi*(1/500)*x + shift) + 1.2 + noise) - np.sqrt(2)
ax3[3].plot(x, y_new, color='black')
ax3[3].set_yticks([])
ax3[3].set_title(r'$\vec{\sigma^2}_2$')

fig3.tight_layout()
plt.savefig('../assets/spectra_row3.pdf', dpi=300)
plt.close(fig3)

# le measure
print((np.linalg.norm(water_spectra - avg_spectra) + 
       np.linalg.norm(water_spectra - so2_yi)) / 2)


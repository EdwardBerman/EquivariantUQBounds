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


if __name__ == "__main__":

    set_rc_params(fontsize=20)

    board = np.array([
        ['X', ' ', 'O'],
        [' ', 'X', ' '],
        [' ', ' ', ' ']
    ])

    fig, ax = plt.subplots(2, 3, figsize=(18, 12))
    
    for i in range(board.shape[0] + 1):
        ax[0,0].plot([i, i], [0, 3], 'k', linewidth=2)
        ax[0,0].plot([0, 3], [i, i], 'k', linewidth=2)
    
    for i in range(board.shape[0]):
        for j in range(board.shape[1]):
            if board[i, j] == 'X':
                ax[0,0].plot([j, j+1], [2-i+1, 1-i+1], 'r', linewidth=2)
                ax[0,0].plot([j, j+1], [1-i+1, 2-i+1], 'r', linewidth=2)
            elif board[i, j] == 'O':
                circle = plt.Circle((j+0.5, 2-i+0.5), 0.4, color='b', fill=False, linewidth=2)
                ax[0,0].add_patch(circle)
    
    ax[0,0].set_xlim(0, 3)
    ax[0,0].set_ylim(0, 3)
    ax[0,0].set_xticks([])
    ax[0,0].set_yticks([])
    ax[0,0].set_aspect('equal')
    ax[0,0].set_title('Tic-tac-toe board')

    board_rotated = np.array([
        [' ', ' ', 'X'],
        [' ', 'X', ' '],
        [' ', ' ', 'O']
    ])
    
    for i in range(board_rotated.shape[0] + 1):
        ax[1,0].plot([i, i], [0, 3], 'k', linewidth=2)
        ax[1,0].plot([0, 3], [i, i], 'k', linewidth=2)
    
    for i in range(board.shape[0]):
        for j in range(board.shape[1]):
            if board_rotated[i, j] == 'X':
                ax[1,0].plot([j, j+1], [2-i+1, 1-i+1], 'r', linewidth=2)
                ax[1,0].plot([j, j+1], [1-i+1, 2-i+1], 'r', linewidth=2)
            elif board_rotated[i, j] == 'O':
                circle = plt.Circle((j+0.5, 2-i+0.5), 0.4, color='b', fill=False, linewidth=2)
                ax[1,0].add_patch(circle)

    ax[1,0].set_xlim(0, 3)
    ax[1,0].set_ylim(0, 3)
    ax[1,0].set_xticks([])
    ax[1,0].set_yticks([])
    ax[1,0].set_aspect('equal')
    ax[1,0].set_title('Rotated Tic-tac-toe board')

    Q_board = np.array([
        [0.0, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.5, 0.5, 1.0]
    ])

    heatmap = ax[0,1].imshow(Q_board, cmap='cool', vmin=0, vmax=1)
    ax[0,1].set_xticks([])
    ax[0,1].set_yticks([])
    ax[0,1].set_title('Q-values')
    #cbar = fig.colorbar(heatmap, ax=ax[0,1], fraction=0.046, pad=0.04)
    #cbar.ax.tick_params(labelsize=16)
    for i in range(Q_board.shape[0]):
        for j in range(Q_board.shape[1]):
            ax[0,1].text(j, i, f'{Q_board[i, j]:.1f}', ha='center', va='center', color='k')

    Q_board_rotated = np.array([
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [1.0, 0.5, 0.0]
    ])

    heatmap = ax[1,1].imshow(Q_board_rotated, cmap='cool', vmin=0, vmax=1)
    ax[1,1].set_xticks([])
    ax[1,1].set_yticks([])
    ax[1,1].set_title('Rotated Q-values')
    #cbar = fig.colorbar(heatmap, ax=ax[1,1], fraction=0.046, pad=0.04)
    #cbar.ax.tick_params(labelsize=16)
    for i in range(Q_board_rotated.shape[0]):
        for j in range(Q_board_rotated.shape[1]):
            ax[1,1].text(j, i, f'{Q_board_rotated[i, j]:.1f}', ha='center', va='center', color='k')


    random_noise = np.random.uniform(0, 0.1, size=(3, 3))

    noise_map = ax[0,2].imshow(random_noise, cmap='coolwarm', vmin=-0.2, vmax=0.2)
    ax[0,2].set_xticks([])
    ax[0,2].set_yticks([])
    ax[0,2].set_title('Confidence')
    #cbar = fig.colorbar(noise_map, ax=ax[0,2], fraction=0.046, pad=0.04)
    #cbar.ax.tick_params(labelsize=16)
    for i in range(random_noise.shape[0]):
        for j in range(random_noise.shape[1]):
            ax[0,2].text(j, i, f'{random_noise[i, j]:.2f}', ha='center', va='center', color='k')

    random_noise_rotated = np.rot90(random_noise, k=-1)

    noise_map = ax[1,2].imshow(random_noise_rotated, cmap='coolwarm', vmin=-0.2, vmax=0.2)
    ax[1,2].set_xticks([])
    ax[1,2].set_yticks([])
    ax[1,2].set_title('Rotated Confidence')
    #cbar = fig.colorbar(noise_map, ax=ax[1,2], fraction=0.046, pad=0.04)
    #cbar.ax.tick_params(labelsize=16)
    for i in range(random_noise_rotated.shape[0]):
        for j in range(random_noise_rotated.shape[1]):
            ax[1,2].text(j, i, f'{random_noise_rotated[i, j]:.2f}', ha='center', va='center', color='k')

    plt.savefig('../assets/tictactoe.pdf', bbox_inches='tight')



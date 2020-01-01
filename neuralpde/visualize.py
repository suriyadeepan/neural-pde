from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.io import loadmat


"""
def figsize(scale, nplots = 1):
  fig_width_pt = 390.0
  inches_per_pt = 1.0/72.27
  golden_mean = (np.sqrt(5.0)-1.0)/2.0
  fig_width = fig_width_pt*inches_per_pt*scale
  fig_height = nplots*fig_width*golden_mean 
  fig_size = [fig_width,fig_height]
  return fig_size


def newfig(width, nplots = 1):
  fig = plt.figure(figsize=figsize(width, nplots))
  ax = fig.add_subplot(111)
  return fig, ax

"""


def savefig(filename, crop = True):
  if crop == True:
    plt.savefig('{}.pdf'.format(filename), bbox_inches='tight', pad_inches=0)
    plt.savefig('{}.eps'.format(filename), bbox_inches='tight', pad_inches=0)
  else:
    plt.savefig('{}.pdf'.format(filename))
    plt.savefig('{}.eps'.format(filename))


def prepare_data_for_plot(bounds, u_pred, v_pred):
  mat = loadmat('data/NLS.mat')
  exact_t, exact_x = mat['t'], mat['x']
  u_pred = u_pred.detach().numpy()
  v_pred = v_pred.detach().numpy()
  exact_u = np.real(mat['usol'])
  exact_v = np.imag(mat['usol'])
  exact_uv = np.sqrt(exact_u**2 + exact_v**2)
  lb = np.array([bounds['t'][0], bounds['x'][0]])
  ub = np.array([bounds['t'][1], bounds['x'][1]])
  t_grid, x_grid = np.meshgrid(exact_t, exact_x)
  t_flat, x_flat = t_grid.reshape(-1, 1), x_grid.reshape(-1, 1)
  tx_flat = np.concatenate([t_flat, x_flat], axis=1)

  return (tx_flat, t_grid, x_grid), exact_uv, (u_pred, v_pred), (lb, ub)

def plot_dynamics(bounds, u_pred, v_pred):
  # prepare data for plotting dynamics
  inputs, groundtruth, predictions, bounds = prepare_data_for_plot(bounds, u_pred, v_pred)
  # get inputs
  tx, t_grid, x_grid = inputs
  # get (exact dynamics) ground truth
  exact_uv = groundtruth
  # get predictions
  u_pred, v_pred = predictions
  # get bounds
  lb, ub = bounds
  # make grid data for *prediction*
  U_pred = griddata(tx, u_pred, (t_grid, x_grid), method='cubic')
  V_pred = griddata(tx, v_pred, (t_grid, x_grid), method='cubic')
  UV_pred = np.sqrt(U_pred**2 + V_pred**2)

  # fig, ax = newfig(1.0, 0.6)

  fig = plt.figure(figsize=(14, 8))
  ax = fig.add_subplot(111)
  ax.axis('off')

  gs = gridspec.GridSpec(1, 2)
  gs.update(top=0.8, bottom=0.2, left=0.1, right=0.9, wspace=0.5)
  ax = plt.subplot(gs[:, 0])
  h = ax.imshow(exact_uv, interpolation='nearest', cmap='jet', 
              extent=[lb[0], ub[0], lb[1], ub[1]],
              origin='lower', aspect='auto')
  divider = make_axes_locatable(ax)
  cax = divider.append_axes("right", size="5%", pad=0.05)

  fig.colorbar(h, cax=cax)
  ax.set_xlabel('$t$')
  ax.set_ylabel('$x$')
  ax.set_title('Exact Dynamics', fontsize=16)
  
  # plot predicted dynamics
  ax = plt.subplot(gs[:, 1])
  h = ax.imshow(UV_pred, interpolation='nearest', cmap='jet', 
              extent=[lb[0], ub[0], lb[1], ub[1]], 
              origin='lower', aspect='auto')
  divider = make_axes_locatable(ax)
  cax = divider.append_axes("right", size="5%", pad=0.05)

  fig.colorbar(h, cax=cax)
  ax.set_xlabel('$t$')
  ax.set_ylabel('$x$')
  ax.set_title('Learned Dynamics', fontsize=16)
  plt.show()

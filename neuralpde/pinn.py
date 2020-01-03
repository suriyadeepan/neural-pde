from neuralpde.nnutils import jacobian
from neuralpde import nnutils

import torch
import torch.nn as nn


class PiDiscoveryNet:

  def __init__(self, layers, bounds):
    self.bounds = bounds
    self.net = nnutils.NeuralNet(layers)
    self.lambda_1 = nn.Parameter(torch.zeros(1,))
    self.lambda_2 = nn.Parameter(torch.zeros(1,))

  def __call__(self, x, y, t):
    # create torch variables from tensors
    x, y, t = nnutils.tvs(x, y, t, shape=(1,))
    # run neural network given (x, y, t)
    psi, p = self.net(torch.cat([x, y, t])).unbind()
    psi, p = psi.view(-1,), p.view(-1,)
    # calculate first derivative
    u, u_y = jacobian(psi, y, hess=True)
    v, v_x = jacobian(psi, x, hess=True)
    u_x, u_xx = jacobian(u, x, hess=True)
    v_y, v_yy = jacobian(v, y, hess=True)
    #   w.r.t `t`
    u_t = jacobian(u, t)
    v_t = jacobian(v, t)
    # remaining calculate second derivatives
    v_xx = jacobian(v_x, x)
    u_yy = jacobian(u_y, y)
    # w.r.t `p`
    p_x = jacobian(p, x)
    p_y = jacobian(p, y)
    # calculate f, g from eqn(6) pg.6
    f_u = u_t + self.lambda_1 * (u * u_x + v * u_y) + \
        p_x - self.lambda_2 * (u_xx + u_yy)
    f_v = v_t + self.lambda_2 * (u * v_x + v * v_y) + \
        p_y - self.lambda_2 * (v_xx + v_yy)

    return u, v, p, f_u, f_v

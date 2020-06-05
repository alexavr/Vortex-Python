# VortexModels
# 1. Simple Model based on VortexModel_Simple.ipynb

import numpy as np
import scipy.stats as st


def vsvm_pressure(X, Y, variance=0.3, center=[0, 0], alpha=1000.):
    """Creates the pressure filed as an boundary condition for VSVM
    """
    pos = np.dstack((X, Y))
    p = 101500. - alpha*st.multivariate_normal.pdf(pos, mean=center, cov=np.eye(2) * variance)
    return p

def vsvm_background_flow(u, v, flow='flat', umax=5):
    """Adds disturbance into existed 2D velocity field
    """

    N = u.shape[0]

    # Disturbing the vortex with
    if flow == 'flat':
        ubg = np.full(N, umax)  # Flat flow
    elif flow == 'couette':
        ubg = np.linspace(0.0, umax, num=N)  # Couette flow at the backgroung
    else:
        print("Cannot recognise flow type!")
        return 0, 0

    u = u + ubg[:, None]
    v = v

    return u, v


def vsvm(pressure2d, dxy):
    """The Very Simple Vortex Model
        Uses pressure fiels as initial 
        and computes velocity according to goesprophic balance 
    """

    f = 10**-4  # midlats
    rho = 1.    # density is constant

    dpdx = np.gradient(pressure2d, dxy, axis=1)  # 1 is row in python
    dpdy = np.gradient(pressure2d, dxy, axis=0)  # 0 is column in python

    u = -1./(f*rho)*dpdy
    v = +1./(f*rho)*dpdx

    return u, v

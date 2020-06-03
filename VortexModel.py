# VortexModels
# 1. Simple Model based on VortexModel_Simple.ipynb

import numpy as np
import scipy.stats as st


def pressure(X, Y, variance=0.3, center=[0, 0], alpha=1000.):
    pos = np.dstack((X, Y))
    p = 101500. - alpha*st.multivariate_normal.pdf(pos, mean=center, cov=np.eye(2) * variance)
    return p


def VortexModel_Simple(pressure2d, dxy, flow='flat', umax=5):

    f = 10**-4 # midlats
    rho = 1.   # density is constant

    N = pressure2d.shape[0]

    # Disturbing the vortex with
    if flow == 'flat':
        ubg = np.full(N, umax)  # Flat flow
    elif flow == 'couette':
        ubg = np.linspace(0.0, umax, num=N)  # Couette flow at the backgroung
    else:
        print("Cannot recognise flow type!")
        return 0, 0

    dpdx = np.gradient(pressure2d, dxy, axis=1) # 1 is row in python
    dpdy = np.gradient(pressure2d, dxy, axis=0) # 0 is column in python

    u = -1./(f*rho)*dpdy + ubg[:, None]
    v = +1./(f*rho)*dpdx

    return u, v

import numpy as np


def vertical_vorticity(u, v, dx, dy):
    dudy = np.gradient(u, dy, axis=0)
    dvdx = np.gradient(v, dx, axis=1)
    return (dvdx-dudy)

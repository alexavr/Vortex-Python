import numpy as np
import scipy.linalg as la

def velocity_gradient_tensor(u=0, v=0, w=0, dx=0, dy=0, dz=0):

    uarray = isinstance(u, np.ndarray)
    varray = isinstance(v, np.ndarray)
    warray = isinstance(w, np.ndarray)

    if uarray:
        dudx = np.gradient(u, dx, axis=1)
        dudy = np.gradient(u, dy, axis=0)
    if varray:
        dvdx = np.gradient(v, dx, axis=1)
        dvdy = np.gradient(v, dy, axis=0)
    if warray:
        dudz = np.gradient(u, dz, axis=0)
        dvdz = np.gradient(v, dz, axis=0)
        dwdx = np.gradient(w, dx, axis=1)
        dwdy = np.gradient(w, dy, axis=0)
        dwdz = np.gradient(w, dz, axis=0)

    if warray:
        res = np.full((u.shape[0], u.shape[1], u.shape[2], 3, 3), -999.)
        res[:, :, :, 0, 0] = dudx
        res[:, :, :, 0, 1] = dudy
        res[:, :, :, 0, 2] = dudz
        res[:, :, :, 1, 0] = dvdx
        res[:, :, :, 1, 1] = dvdy
        res[:, :, :, 1, 2] = dvdz
        res[:, :, :, 3, 0] = dwdx
        res[:, :, :, 3, 1] = dwdy
        res[:, :, :, 3, 2] = dwdz
    else:
        res = np.full((u.shape[0], u.shape[1], 2, 2), -999.)
        res[:, :, 0, 0] = dudx
        res[:, :, 0, 1] = dudy
        res[:, :, 1, 0] = dvdx
        res[:, :, 1, 1] = dvdy

    return res


def q(velocity_gradient_tensor):
    dims = len(velocity_gradient_tensor.shape)

    if dims == 4:
        dim1 = 2
        dim2 = 3
    else:
        dim1 = 3
        dim2 = 4

    tgradvt = np.swapaxes(velocity_gradient_tensor, dim1, dim2)

    omega = 0.5*(velocity_gradient_tensor - tgradvt)
    symmetric = 0.5*(velocity_gradient_tensor + tgradvt)

    omega_norm = np.linalg.norm(omega, axis=(dim1, dim2))
    symmetric_norm = np.linalg.norm(symmetric, axis=(dim1, dim2))

    q = 0.5*(omega_norm**2 - symmetric_norm**2)

    return q


def lambda2(tgradv):
    dims = len(tgradv.shape)

    if dims == 4:
        dim1 = 2
        dim2 = 3
    else:
        dim1 = 3
        dim2 = 4

    tgradvt = np.swapaxes(tgradv, dim1, dim2)

    o2 = (0.5*(tgradv - tgradvt))**2
    s2 = (0.5*(tgradv + tgradvt))**2

    A = s2 + o2

    L = la.eigvals(a)



    # l2 = 0.5*(omega_norm**2 - symmetric_norm**2)

    return l2

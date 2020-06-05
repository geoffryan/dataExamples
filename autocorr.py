""" Cribbed exactly from DFM's emcee v3"""

import numpy as np

def next_pow_two(n):
    i=1
    while i < n:
        i = i << 1
    return i

def function_1d(x):

    x = np.atleast_1d(x)
    if len(x.shape) != 1:
        raise ValueError("x gotta be 1D man")
    n = next_pow_two(len(x))

    f = np.fft.fft(x - np.mean(x), n=2*n)
    acf = np.fft.ifft(f * np.conjugate(f))[:len(x)].real
    acf /= acf[0]
    return acf


def auto_window(taus, c):
    m = np.arange(len(taus)) < c * taus
    if np.any(m):
        return np.argmin(m)
    return len(taus)-1


def integrated_time(x, c=5, tol=50, quiet=False, timeAxis=1,
                    walkerAxis=0):

    x = np.atleast_1d(x)

    if len(x.shape) == 1:
        x = x[None, :]

    if (timeAxis < 0 or timeAxis >= len(x.shape)
            or walkerAxis < 0 or walkerAxis >= len(x.shape)):
        raise ValueError("invalid dimensions and axis choices!")

    n_t = x.shape[timeAxis]
    n_w = x.shape[walkerAxis]

    x_ord = np.moveaxis(x, (timeAxis, walkerAxis), (-1, -2))
    shape_ord = x_ord.shape[:-2]

    tau_est = np.empty(shape_ord)
    windows = np.empty(shape_ord, dtype=int)
    n_d = np.prod(shape_ord)

    # Loop over parameters
    for d in range(n_d):
        ind = np.unravel_index(d, shape_ord)
        f = np.zeros(n_t)
        for k in range(n_w):
            f += function_1d(x_ord[ind+(k,)])
        f /= n_w
        taus = 2.0 * np.cumsum(f) - 1.0
        windows[ind] = auto_window(taus, c)
        tau_est[ind] = taus[windows[ind]]

    flag = tol * tau_est > n_t

    if np.any(flag):
        if not quiet:
            raise ValueError("Chain too short! "
                             + "Longest tau ({0:.0f}) ".format(tau_est.max())
                             + "* tol ({0:.0f}) (={1:.0f})".format(tol,
                             tol*tau_est.max())
                             + "is longer than  n_t ({0:d})!".format(n_t))

    return tau_est

import numpy as np
import scipy
import scipy.interpolate
import scipy.special


def whiten_transforms_from_cdf(x, cdf):
    """
    Calculate a pair of transforms to whiten and unwhiten a distribution.
    The whitening transform is monotonic and invertible.
    Parameters
    ----------
    x : array
        1D array of non-decreasing values giving bin edges for the distribution
        to whiten and unwhiten.
    cdf : array
        1D array of non-decreasing values giving the cummulative probability
        density associated with each bin edge.  Does not need to be normalized.
        Must have the same length as x.
    Returns
    -------
    tuple
        Tuple (F,G) of callable objects that whiten y=F(x) and unwhiten x=G(y)
        samples x of the input distribution, so that y has a Gaussian
        distribution with zero mean and unit variance.
    """
    x = np.asarray(x)
    cdf = np.asarray(cdf)
    if x.shape != cdf.shape:
        raise ValueError('Input arrays must have same shape.')
    if len(x.shape) != 1:
        raise ValueError('Input arrays must be 1D.')
    if not np.all(np.diff(x) >= 0):
        raise ValueError('Values of x must be non-decreasing.')
    if not np.all(np.diff(cdf) >= 0):
        raise ValueError('Values of cdf must be non-decreasing.')
    # Normalize.
    cdf /= cdf[-1]
    # Use linear interpolation for the forward and inverse transforms between
    # the input range and Gaussian CDF values.
    args = dict(
        kind='linear', assume_sorted=True, copy=False, bounds_error=True)
    forward = scipy.interpolate.interp1d(x, cdf, **args)
    backward = scipy.interpolate.interp1d(cdf, x, **args)
    # Add wrappers to convert between CDF and PDF samples.
    root2 = np.sqrt(2)
    forward_transform = (
        lambda x: root2 * scipy.special.erfinv(2 * forward(x) - 1))
    inverse_transform = (
        lambda y: backward(0.5 * (1 + scipy.special.erf(y / root2))))
    return forward_transform, inverse_transform

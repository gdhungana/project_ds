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

def whiten_transforms(data, data_min=None, data_max=None):
    """Calculate a pair of transforms to whiten and unwhiten a distribution.
    Uses :func:`desimodel.weather.whiten_transforms_from_cdf`.
    Parameters
    ----------
    data : array
        1D array of samples from the distribution to whiten.
    data_min : float or None
        Clip the distribution to this minimum value, or at min(data) if None.
        Must be <= min(data).
    data_max : float or None
        Clip the distribution to this maximum value, or at max(data) if None.
        Must be >= max(data).
    Returns
    -------
    tuple
    """
    n_data = len(data)
    # Sort the input data with padding at each end for the min/max values.
    sorted_data = np.empty(shape=n_data + 2, dtype=data.dtype)
    sorted_data[1:-1] = np.sort(data)
    if data_min is None:
        sorted_data[0] = sorted_data[1]
    else:
        if data_min > sorted_data[1]:
            raise ValueError('data_min > min(data)')
        sorted_data[0] = data_min
    if data_max is None:
        sorted_data[-1] = sorted_data[-2]
    else:
        if data_max < sorted_data[-2]:
            raise ValueError('data_max < max(data)')
        sorted_data[-1] = data_max
    # Calculate the Gaussian CDF value associated with each input value in
    # sorted order. The pad values are associated with CDF = 0, 1 respectively.
    cdf = np.arange(n_data + 2) / (n_data + 1.)
    return whiten_transforms_from_cdf(sorted_data, cdf)

def sample_timeseries(x_grid, pdf_grid, psd, n_sample, dt_sec=180., gen=None):
    """Sample a time series specified by a power spectrum and 1D PDF.
    The PSD should describe the temporal correlations of whitened samples.
    Generated samples will then be unwhitened to recover the input 1D PDF.
    See DESI-doc-3087 for details.
    Uses :func:`whiten_transforms_from_cdf`.
    Parameters
    ----------
    x_grid : array
        1D array of N increasing grid values covering the parameter range
        to sample from.
    pdf_grid : array
        1D array of N increasing PDF values corresponding to each x_grid.
        Does not need to be normalized.
    psd : callable
        Function of frequency in 1/days that returns the power-spectral
        density of whitened temporal fluctations to sample from. Will only be
        called for positive frequencies.  Normalization does not matter.
    n_sample : int
        Number of equally spaced samples to generate.
    dt_sec : float
        Time interval between samples in seconds.
    gen : np.random.RandomState or None
        Provide an existing RandomState for full control of reproducible random
        numbers, or None for non-reproducible random numbers.
    """
    x_grid = np.array(x_grid)
    pdf_grid = np.array(pdf_grid)
    if not np.all(np.diff(x_grid) > 0):
        raise ValueError('x_grid values are not increasing.')
    if x_grid.shape != pdf_grid.shape:
        raise ValueError('x_grid and pdf_grid arrays have different shapes.')
    # Initialize random numbers if necessary.
    if gen is None:
        gen = np.random.RandomState()
    # Calculate the CDF.
    cdf_grid = np.cumsum(pdf_grid)
    cdf_grid /= cdf_grid[-1]
    # Calculate whitening / unwhitening transforms.
    whiten, unwhiten = whiten_transforms_from_cdf(x_grid, cdf_grid)
    # Build a linear grid of frequencies present in the Fourier transform
    # of the requested time series.  Frequency units are 1/day.
    dt_day = dt_sec / (24. * 3600.)
    df_day = 1. / (n_sample * dt_day)
    f_grid = np.arange(1 + (n_sample // 2)) * df_day
    # Tabulate the power spectral density at each frequency.  The PSD
    # describes seeing fluctuations that have been "whitened", i.e., mapped
    # via a non-linear monotonic transform to have unit Gaussian probability
    # density.
    psd_grid = np.empty_like(f_grid)
    psd_grid[1:] = psd(f_grid[1:])
    # Force the mean to zero.
    psd_grid[0] = 0.
    # Force the variance to one.
    psd_grid[1:] /= psd_grid[1:].sum() * df_day ** 2
    # Generate random whitened samples.
    n_psd = len(psd_grid)
    x_fft = np.ones(n_psd, dtype=complex)
    x_fft[1:-1].real = gen.normal(size=n_psd - 2)
    x_fft[1:-1].imag = gen.normal(size=n_psd - 2)
    x_fft *= np.sqrt(psd_grid) / (2 * dt_day)
    x_fft[0] *= np.sqrt(2)
    x = np.fft.irfft(x_fft, n_sample)
    # Un-whiten the samples to recover the desired 1D PDF.
    x_cdf = 0.5 * (1 + scipy.special.erf(x / np.sqrt(2)))
    return np.interp(x_cdf, cdf_grid, x_grid)


#-- sampling example using DESI seeing
#- generate a distribution 
def norm_model(x):
    p = np.array([1.,1.]) #- Gaussian parm for mean=1, std=1.
    y = norm.pdf(x,p[0],p[1])
    return y / (y.sum() * np.gradient(x))




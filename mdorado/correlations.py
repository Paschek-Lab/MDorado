import numpy as np
from scipy import signal

def correlate(a, b=None):
    """
    mdorado.correlations.correlate(a,b=None)

    Calculates the correlation function <a(0)b(t)> from two arrays or
    the autocorrelation function <a(0)a(t)> if just one argument is
    given

    Parameters
    ----------
        a: one-dimensional list or array

        b: one-dimensional list or array, optional
            if not given, the autocorrelation function <a(0)a(t)> is 
            calculated

    Returns
    -------
        ndarray
            correlation function <a(0)b(t)>
    """
    a = np.array(a)
    #If no b is given: compute autocorrelation
    if np.array(b).all() == None:
        b = a
    else:
        b = np.array(b)

    #Error if shape is not 1D
    if len(a.shape) > 1 or len(b.shape) > 1:
        raise TypeError("""
            correlate: a and b have to be one-dimensional arrays or lists.
            """)

    ct = signal.correlate(b, a, method='auto', mode='full')
    #remove values for t<0
    ct = ct[ct.size // 2:]
    ct = ct.astype(float)
    alen = len(a)
    #normalize ct based on number of points for correlation
    ct /= np.flip(np.arange(1, alen+1))
    return ct

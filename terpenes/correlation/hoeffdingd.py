import numpy
import numpy as np
from scipy.stats import rankdata
from sklearn.preprocessing import KBinsDiscretizer


def hoeffdingd(x: numpy.ndarray, y: numpy.ndarray, discretize: bool = False,
               n_bins: int = 50, strategy: str = "uniform") -> float:
    """ Hoeffding's measure of dependence, D (Hoeffding Dependence Coefficient)

    NumPy + SciPy + sklearn implementation

    From formula:
    https://support.sas.com/documentation/cdl/en/procstat/63104/HTML/default/viewer.htm#procstat_corr_sect016.htm

    Modified from:
    https://github.com/PaulVanDev/HoeffdingD/blob/master/EfficientHoeffdingD.ipynb

    Args:
        x (np.array): first data series, 1D shape (n_samples,)
        y (np.array): second data series, 1D shape (n_samples,)
        discretize (bool, default False): if True (continuous data, not
            categorical), discretizes data into bins; default False
            (categorical data)
        n_bins (int, default 50): if discretize == True, discretizes data with
            this many bins
        strategy (str, default 'uniform'): if discretize == True,
            sklearn.preprocessing.KBinsDiscretizer uses strategy in
            {'uniform', 'quantile', 'kmeans'}

    Returns:
        float: Hoeffding Dependence Coefficient
    """
    
    if x.shape != y.shape:
        raise ValueError(f"x and y must be the same shape: " +
                         "{x.shape}, {y.shape}")
    
    if discretize:
        disc = KBinsDiscretizer(n_bins=n_bins, encode="ordinal",
                                strategy=strategy)
        _x = x.reshape(-1, 1)
        disc.fit(_x)
        x = disc.transform(_x)
        disc = KBinsDiscretizer(n_bins=n_bins, encode="ordinal",
                                strategy=strategy)
        _y = _y.reshape(-1, 1)
        disc.fit(_y)
        y = disc.transform(_y)

    R: numpy.ndarray = rankdata(x)
    S: numpy.ndarray = rankdata(y)
    n: int = x.shape[0]
    Q: numpy.ndarray = np.ones(n)

    for i in range(n):

        lt_r = np.array([R[j] < R[i] for j in range(len(R))])
        eq_r = np.array([R[j] == R[i] for j in range(len(R))])
        lt_s = np.array([S[j] < S[i] for j in range(len(S))])
        eq_s = np.array([S[j] == S[i] for j in range(len(S))])
        Q[i] += sum(lt_r & lt_s)
        Q[i] += 1/4 * (sum(eq_r & eq_s) - 1)
        Q[i] += 1/2 * sum(eq_r & lt_s)
        Q[i] += 1/2 * sum(lt_r & eq_s)

    D1 = np.sum(np.multiply((Q - 1), (Q - 2)))
    D2 = np.sum(np.multiply(np.multiply((R - 1), (R - 2)),
                            np.multiply((S - 1), (S - 2))))
    D3 = np.sum(np.multiply(np.multiply((R - 2), (S - 2)), (Q - 1)))

    return 30 * ((n - 2) * (n - 3) * D1 + D2 - 2 * (n - 2) * D3) \
        / (n * (n - 1) * (n - 2) * (n - 3) * (n - 4))

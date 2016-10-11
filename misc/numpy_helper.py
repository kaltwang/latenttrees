import numpy as np
import warnings
import time
from misc.python_helper import isequal_or_none

# optional imports:
try:
    import dirichlet  # from https://github.com/ericsuh/dirichlet
except ImportError:
    pass
try:
    from sklearn.metrics import roc_auc_score
except ImportError:
    pass

class random_getset_state(object):
    def __init__(self, seed=None):
        """
        If there are decorator arguments, the function
        to be decorated is not passed to the constructor!
        """
        self.seed = seed

    def __call__(self, f):
        """
        If there are decorator arguments, __call__() is only called
        once, as part of the decoration process! You can only give
        it a single argument, which is the function object.
        """
        def wrapped_f(*args, **kwargs):
            if self.seed is not None:
                state_before = np.random.get_state()
                np.random.seed(self.seed)
            else:
                state_before = None
            result = f(*args, **kwargs)
            if state_before is not None:
                np.random.set_state(state_before)
            return result
        return wrapped_f


def normalize_mean_std(X, x_mean=None, x_std=None, axis=0):
    if x_mean is None:
        x_mean = np.nanmean(X, axis=axis, keepdims=True)
    if x_std is None:
        x_std = np.nanstd(X, axis=axis, keepdims=True)
        x_std[~np.isfinite(1 / x_std)] = 1

    X = X - x_mean
    X = X / x_std

    return X, x_mean, x_std

def normalize_mean_std_obj_array(X, x_mean=None, x_std=None, axis=0):
    if is_obj_array(X):
        x_mean_out = np.zeros_like(X, dtype=float)
        x_std_out = np.ones_like(X, dtype=float)
        Xr = X.flat
        xmr = x_mean_out.flat
        xsr = x_std_out.flat
        for i in range(X.size):
            if x_mean is None:
                x_mean_act = None
            else:
                x_mean_act = x_mean.ravel()[i]
            if x_std is None:
                x_std_act = None
            else:
                x_std_act = x_std.ravel()[i]
            Xr[i], xmr[i], xsr[i] = normalize_mean_std(Xr[i], x_mean=x_mean_act, x_std=x_std_act, axis=axis)
    else:
        X, x_mean_out, x_std_out = normalize_mean_std(X, x_mean=x_mean, x_std=x_std, axis=axis)

    return X, x_mean_out, x_std_out

def normalize_mean_std_inv(X, x_mean=None, x_std=None):
    if x_std is not None:
        X = X * x_std
    if x_mean is not None:
        X = X + x_mean
    return X

def normalize_mean_std_inv_obj_array(X, x_mean=None, x_std=None):
    if is_obj_array(X):
        Xr = X.flat
        for i in range(X.size):
            if x_mean is None:
                x_mean_act = None
            else:
                x_mean_act = x_mean.ravel()[i]
            if x_std is None:
                x_std_act = None
            else:
                x_std_act = x_std.ravel()[i]
            Xr[i] = normalize_mean_std_inv(Xr[i], x_mean=x_mean_act, x_std=x_std_act)
    else:
        X = normalize_mean_std_inv(X, x_mean=x_mean, x_std=x_std)
    return X


def corr_pairwise(x, y, axis=0, keepdims=False, ddof=0):
    """
    :param x: N x M
    :param y: N x M
    :return: 1 x M
    """
    mean_x = np.mean(x, axis=axis, keepdims=True)
    mean_y = np.mean(y, axis=axis, keepdims=True)
    N = np.prod(x.shape[axis])
    cov = np.sum((x-mean_x) * (y-mean_y), axis=axis, keepdims=keepdims) / (N-ddof)
    std_x = np.std(x, axis=axis, keepdims=keepdims, ddof=ddof)
    std_y = np.std(y, axis=axis, keepdims=keepdims, ddof=ddof)
    corr = cov / (std_x * std_y)
    return corr

def icc_pairwise(x, y, axis=0, keepdims=False, ddof=0):
    # ICC(3,1) (P. E. Shrout and J. L. Fleiss. Intraclass correlations: uses in assessing rater reliability. Psychol. Bull., 86(2):420–428, 1979.)
    # ICC(C,1) Case 3 (K. O. McGraw and S. P. Wong. Forming inferences about some intraclass correlation coefficients. Psychol. Methods, 1(1):30–46, 1996.)
    mean_x = np.mean(x, axis=axis, keepdims=True)
    mean_y = np.mean(y, axis=axis, keepdims=True)
    N = np.prod(x.shape[axis])
    cov = np.sum((x - mean_x) * (y - mean_y), axis=axis, keepdims=keepdims) / (N - ddof)
    var_x = np.var(x, axis=axis, keepdims=keepdims, ddof=ddof)
    var_y = np.var(y, axis=axis, keepdims=keepdims, ddof=ddof)
    icc = 2 * cov / (var_x + var_y)
    return icc

def roc_auc_score_safe(truth, pred):
        auc = []
        for i in range(truth.shape[1]):
            if len(np.unique(truth[:, i])) != 2:
                auc.append(np.nan)
            else:
                auc.append(roc_auc_score(truth[:, i], pred[:, i], average=None))
        return np.array(auc)

def downsample_idx(N, N_max=100, axis=0, method='equidist'):
    if N > N_max:
        if method == 'equidist':
            step = (N - 1) / N_max
            idx_cont = (np.arange(N_max) + 0.5) * step

            # add small slope to idx-cont, to avoid rounding neighbouring values to the same integer.
            # max absolute value added/subtracted is 1/10 of the step size
            adjust = ((idx_cont * 2 / (N - 1)) - 1) * step / 10
            idx_cont += adjust

            idx = np.array(np.round(idx_cont), dtype=int)

        if method == 'random':
            idx = np.random.choice(N, size=N_max, replace=False)
            idx = np.sort(idx)
    else:
        idx = np.s_[:]
    return idx

def downsample(x, N_max=100, axis=0, method='equidist'):
    N = x.shape[axis]
    idx = downsample_idx(N, N_max=N_max, axis=axis, method=method)
    idx_all = [np.s_[:]] * x.ndim
    idx_all[axis] = idx
    x = x[idx_all]
    return x

def get_chunks(x, nbytes_desired):
    nbytes = np.array(x).ravel()[0].nbytes
    size_desired = nbytes_desired / nbytes
    if size_desired >= x.size:
        # desired chunk size is greater or equal than array size, thus we can include the whole array in a single chunk
        return x.shape
    s = x.shape[::-1]
    cp = np.cumprod(s)
    dim = np.argmax(cp >= size_desired)
    s_dim_desired = size_desired / np.prod(s[:dim])
    s_dim = np.round(s_dim_desired)
    if s_dim < 1:
        s_dim = 1
    chunks = np.ones_like(s)
    chunks[:dim] = s[:dim]
    chunks[dim] = s_dim
    result = tuple(chunks[::-1])
    return result


def normalize_convex(x, axis=None):
    """Normalizes x in-place and returns the normalizing constant s
    :param x: numpy float
    :param axis: axis to normalize over
    """
    s = np.sum(x, axis=axis, keepdims=True)
    assert np.all(s != 0)
    x /= s  # will fail if x is not of type float
    return s

def obj_array_get_N(a):
    if is_obj_array(a):
        N = a.ravel()[0].shape[0]
    else:
        N = a.shape[0]
    return N

def is_obj_array(a):
    result = a.dtype == np.dtype('O') and a.ndim >= 1 and a.shape[0] == 1
    return result

def to_obj_array(X):
    M = X.shape[1]
    Xo = np.empty((1, M), dtype=object)
    for i in range(M):
        Xo[0, i] = X[:, i]
    return Xo

def from_obj_array(Xo):
    M = Xo.shape[1]
    f = Xo[0,0]
    s = list(f.shape)
    s_new = s[:1] + [M] + s[1:]
    X = np.empty(s_new, dtype=f.dtype)
    for i in range(M):
        X[:, i] = Xo[0, i]
    return X

class ProgressLine(object):
    def __init__(self, prefix='', time_min=1):
        self.prefix = prefix
        self.time_min = time_min

        self.__perc = 0
        self.__time_last = time.time()
        self.__first_print = True

    def progress(self, perc):
        now = time.time()
        if (perc > self.__perc) and (now - self.__time_last > self.time_min):
            self.__perc = perc
            self.__time_last = now
            str_ = " {}".format(perc)
            if self.__first_print:
                str_ = self.prefix + "(%):" + str_
                self.__first_print = False
            print(str_, end='', flush=True)

    def finish(self):
        if not self.__first_print:
            print("")


def obj_array_concatenate(oa):
    so = oa.shape  # (10, 2, 9)

    d0 = np.prod(so[:-1])
    d1 = so[-1]
    oa = oa.reshape((d0, d1))

    result = []
    for i in range(d0):
        oa_act = oa[i,:]
        result_act = np.concatenate(oa_act, axis=0)
        result.append(result_act)

    result = np.stack(result, axis=0)
    si = result.shape[1:]
    result = result.reshape(so[:-1] + si)
    # result.shape = (10, 2, 9*14535, 12)
    return result

def randomize_obj_array(X, K):
    if is_obj_array(X):
        X = X[0]
        obj_array = True
    else:
        obj_array = False

    X = randomize_select(X, K)

    if obj_array:
        Xo = np.empty((1,), dtype=object)
        Xo[0] = X
        X = Xo

    return X


def randomize_select(X, K):
    if K == 1:
        assert X.ndim == 1
        # X.shape == (N,)
        X = randomize_gaussian(X)
    else:
        if X.ndim > 1:
            assert X.shape[1] == K
            # X.shape == (N,K)
            X = randomize_dirichlet(X)
        else:
            assert X.ndim == 1
            # X.shape == (N,)
            X = randomize_categorical(X, K)
    return X

def randomize_gaussian(X):
    mean = np.mean(X)
    std = np.std(X)
    if std > 0:
        X = np.random.normal(loc=mean, scale=std, size=X.shape)
    else:
        warnings.warn('std is zero, thus leave X constant')
    return X

def randomize_categorical(X, K):
    unique, counts = np.unique(X, return_counts=True)
    p = np.zeros((K,))
    sum = np.sum(counts)
    for i in range(len(unique)):
        u = unique[i]
        c = counts[i]
        p[int(u)] = c / sum

    X = np.random.choice(K, size=X.shape, p=p)
    return X

def randomize_dirichlet(X):
    # additive smoothing to avoid numerical problems
    M = X.shape[1]
    # eps = np.spacing(1)
    eps = 0.01 / M
    X = (X + eps) / (1 + M * eps)
    alpha = dirichlet.mle(X)
    N = X.shape[0]
    X = np.random.dirichlet(alpha, size=N)
    return X

def has_equiv_shape(x, required_shape):
    """Asserts that the numpy array x has the required shape. Required dimensions of None are ignored.
    E.g. x.shape=(3,5,6) is equivalent to required_shape=(3,None,6)
    :param x: numpy array
    :param required_shape: tuple of required shape dimensions
    """
    required_ndim = len(required_shape)
    ndim = x.ndim
    if required_ndim != ndim:
        return False
    shape = x.shape
    for d in range(ndim):
        if not isequal_or_none(required_shape[d], shape[d]):
            return False
    return True

def cut_max(x, num_max=None):
    if num_max is not None:
        num = min(x.shape[0], num_max)
        return x[0:num,:]
    else:
        return x

# this is ~100 times faster than scipy.norm(mu, std).logpdf(x)
LOG_SQRT_2PI = np.log(np.sqrt(2*np.pi))
def norm_logpdf(x, mu=0, std=1):
    x = (x - mu) / std
    logpdf = -x*x/2 - LOG_SQRT_2PI - np.log(std)
    return logpdf

def normalize_convex_log(x, axis=None):
    """Normalizes x_log in-place and returns the normalizing constant s_log. The returned x will not be in log space
    anymore. s is returned in log space.
    :param x: numpy float
    :param axis: axis to normalize over
    """
    m = np.amax(x, axis=axis, keepdims=True)
    x -= m
    np.exp(x, out=x)
    s = normalize_convex(x, axis=axis)
    s_log = m + np.log(s)
    return s_log

def expand_array(x, shape, constant_value=np.nan):
    arr_s = np.array(shape)
    arr_xs = np.array(x.shape)
    arr_pad = np.maximum(arr_s - arr_xs, 0)
    zeros = np.zeros_like(arr_pad)
    pad = np.concatenate((zeros[:,np.newaxis], arr_pad[:,np.newaxis]), axis=1)
    x = np.pad(x, pad, mode='constant', constant_values=(constant_value,))
    return x
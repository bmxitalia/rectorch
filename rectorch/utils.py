r"""Module containing utility functions.
"""
import os
import json
from torch.optim import Adam, SGD, Adagrad, Adadelta, Adamax, AdamW
import torch
import cvxopt as co
import numpy as np
from scipy.sparse import csr_matrix
import rectorch

# AUTHORSHIP
__version__ = "0.9.0dev"
__author__ = "Mirko Polato"
__copyright__ = "Copyright 2020, rectorch"
__license__ = "MIT"
__maintainer__ = "Mirko Polato"
__email__ = "mak1788@gmail.com"
__status__ = "Development"
#

__all__ = ['init_optimizer', 'get_data_cfg', 'prepare_for_prediction', 'tensor_apply_permutation',
           'cvxopt_diag', 'md_kernel', 'kernel_normalization']

def init_optimizer(params, opt_cfg=None):
    r"""Get a new optimizer initialize according to the given configurations.

    Parameters
    ----------
    params: iterable
        Iterable of parameters to optimize or dicts defining parameter groups.
    opt_cfg : :obj:`dict` or :obj:`None` [optional]
        Dictionary containing the configuration for the optimizer, by default :obj:`None`.
        If :obj:`None` a default optimizer is returned, i.e., ``torch.optim.Adam(params)``.
    """
    if opt_cfg is None:
        return Adam(params)

    cfg = opt_cfg.copy()
    if "name" in cfg:
        opt_name = cfg['name']
        del cfg['name']
    else:
        opt_name = "adam"

    if opt_name == "adam":
        opt_cls = Adam
    elif opt_name == "adadelta":
        opt_cls = Adadelta
    elif opt_name == "adagrad":
        opt_cls = Adagrad
    elif opt_name == "adamw":
        opt_cls = AdamW
    elif opt_name == "adamax":
        opt_cls = Adamax
    elif opt_name == "sgd":
        opt_cls = SGD

    return opt_cls(params, **cfg)

def get_data_cfg(ds_name=None):
    r"""Return standard data processing/splitting configuration.

    Parameters
    ----------
    ds_name : :obj:`str` [optional]
        The name of the dataset. Possible values are:

        * 'ml100k' : Movielens 100k;
        * 'ml1m' : Movielens 1 million;
        * 'ml20m': Movielens 20 million;
        * 'msd' : Million Song Dataset;
        * 'netflix' : Netflix Challenge Dataset;
        * 'ml100k_ncr' : Movielens 100k for NCR model.

        .. warning:: The Netflix dataset is assumed of being merged into a single CSV file named
           'ratings.csv'. In the `github homepage <https://github.com/makgyver/rectorch>`_
           of the framework it is provided a python notebook (``process_netflix.ipynb``) that
           performs such merging.

        If :obj:`None` the function returns a generic configuration with no thresholding and
        horizontal leave-one-out splitting. The name of the raw file is empty and must be set.

    Returns
    -------
    :obj:`dict`
        Dictionary containing the configurations.
    """
    p = os.path.dirname(rectorch.__file__) + "/"
    if ds_name == "ml100k":
        with open(p + 'config/config_data_ml100k.json') as f:
            cfg = json.load(f)
    elif ds_name == "ml1m":
        with open(p + 'config/config_data_ml1m.json') as f:
            cfg = json.load(f)
    elif ds_name == "ml20m":
        with open(p + 'config/config_data_ml20m.json') as f:
            cfg = json.load(f)
    elif ds_name == "netflix":
        with open(p + 'config/config_data_netflix.json') as f:
            cfg = json.load(f)
    elif ds_name == "msd":
        with open(p + 'config/config_data_msd.json') as f:
            cfg = json.load(f)
    elif ds_name == "ml100k_ncr":
        with open(p + 'config/config_data_ml100k_ncr.json') as f:
            cfg = json.load(f)
    else:
        cfg = {
            "processing": {
                "data_path": None,
                "threshold": 0,
                "separator": ",",
                "header": None,
                "u_min": 0,
                "i_min": 0
            },
            "splitting": {
                "split_type": "horizontal",
                "sort_by": None,
                "seed": 42,
                "shuffle": False,
                "valid_size": 1,
                "test_size": 1,
                "test_prop": .5
            }
        }

    return cfg


def sparse2tensor(sparse_matrix):
    """Convert a scipy.sparse.csr_matrix to torch.FloatTensor

    Parameters
    ----------
    sparse_matrix : :class:`scipy.sparse.csr_matrix`
        The matrix to convert.

    Returns
    -------
    :class:`torch.FloatTensor`
        The converted matrix.
    """
    coo = sparse_matrix.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))
    return torch.sparse.FloatTensor(torch.LongTensor(indices),
                                    torch.FloatTensor(values),
                                    torch.Size(coo.shape)).to_dense()


def prepare_for_prediction(data_input, ground_truth):
    r"""Prepare the data for performing prediction.

    Parameters
    ----------
    data_input : any type
        The input data for prediction
    ground_truth : any type
        The ground truth data.

    Returns
    -------
    :obj:`tuple`
        The tuple containing input data and the ground truth.

    Raises
    ------
    :class:`ValueError`
        Raised when the input type is not recognized.
    """
    if isinstance(data_input, (tuple, torch.FloatTensor)) and isinstance(ground_truth, csr_matrix):
        return data_input, ground_truth.toarray()
    elif isinstance(data_input, (torch.FloatTensor, torch.LongTensor)):
        data_input = data_input.view(data_input.shape[0], -1)
        ground_truth = ground_truth.view(ground_truth.shape[0], -1).cpu().numpy()
        return (data_input,), ground_truth
    elif isinstance(data_input, tuple):
        return data_input, ground_truth
    else:
        raise ValueError("Unrocognize 'data_input' type.")


def tensor_apply_permutation(x, permutation):
    r"""Apply a indices premutation tensor to a 2D tensor.

    Parameters
    ----------
    x : :class:`torch.Tensor`
        The tensor to permute.
    permutation : :class:`torch.Tensor`
        The tensor containing the rows' permutation indices.

    Returns
    -------
    :class:`torch.Tensor`
        The permuted tensor.
    """
    d1, d2 = x.size()
    ret = x[
        torch.arange(d1).unsqueeze(1).repeat((1, d2)).flatten(),
        permutation.flatten()
    ].view(d1, d2)
    return ret


def collect_results(results):
    r"""Collect the results from a results' dictionary.

    The results' dictionary ``results`` contains for each metric (i.e., key) an array of values
    (i.e., metric value for each user). The function compute the average and standard deviation of
    each metric over the users.

    Parameters
    ----------
    results : :obj:`dict` (:obj:`str` - :class:`numpy.ndarray`)
        The results' dictionary.

    Returns
    -------
    :obj:`dict` (:obj:`str`, :obj:`tuple` (:obj:`float, :obj:`float`))
        The mean and standard deviation of each metric.
    """
    return {met : (np.mean(results[met]), np.std(results[met])) for met in results}


def cvxopt_diag(vec):
    r"""Build a CVXopt diagonal matrix.

    Parameters
    ----------
    vec : :class:`cvxopt.matrix`
        The vector to put in the diagonal of the matrix.

    Returns
    -------
    :class:`cvxopt.matrix`
        The diagonal matrix with the vector ``vec`` as diagonal.
    """
    result = co.matrix(0.0, (vec.size[0], vec.size[0]))
    for i, x in enumerate(vec):
        result[i, i] = x
    return result


def md_kernel(X, degree=2):
    r"""Monotone disjunctive kernel (mD-kernel).

    Returns the (boolean) monotone disjunctive kernel of degree ``d``. Given two n-dimensional
    binary vectors :math:`\mathbf{x}` and :math:`\mathbf{z}` the mD-kernel of degree :math:`d` is
    defined as

    :math:`K_\vee^d (\mathbf{x}, \mathbf{z}) = \binom{n}{d} - \binom{n - \|\mathbf{x}\|_1}{d} -\
    \binom{n - \|\mathbf{z}\|_1}{d} + \binom{n - \|\mathbf{x}\|_1 - \|\mathbf{z}\|_1 -\
    \mathbf{x}^\top \mathbf{z}}{d}`.

    Parameters
    ----------
    X : :class:`numpy.ndarray`
        The examples matrix.
    degree : :obj:`int` [optional]
        The degree (integer >= 1) of the mD-kernel, default 2.

    Returns
    -------
    :class:`numpy.ndarray`
        The monotone disjunctive kernel.
    """
    def md_kernel_gen(K0, n, degree=2):
        N = np.full(K0.shape, n)
        XX = np.dot(np.diag(K0).reshape(K0.shape[0], 1), np.ones((1, K0.shape[0])))
        N_x = N - XX
        N_xz = N_x - XX.T + K0
        N_d, N_xd, N_xzd = N.copy(), N_x.copy(), N_xz.copy()

        yield N_d - N_xd - N_xd.T + N_xzd
        for d in range(1, degree):
            N_d = N_d * (N - d) / (d + 1)
            N_xd = N_xd * (N_x - d) / (d + 1)
            N_xzd = N_xzd * (N_xz - d) / (d + 1)
            yield N_d - N_xd - N_xd.T + N_xzd

    assert degree >= 1 and isinstance(degree, int), "'degree' must be an integer >= 1"
    mdk = None
    for ki in md_kernel_gen(np.dot(X.T, X), X.shape[0], degree):
        mdk = ki

    return mdk


def kernel_normalization(K):
    r"""Apply the kernel normalization.

    Given the kernel :math:`\mathbf{K}` its normalized version can be computed as

    :math:`\tilde{\mathbf{K}} = \frac{\mathbf{K}}{\sqrt{\mathbf{d}^\top \mathbf{d}}}`

    where :math:`\mathbf{d}` is the diagonal (vector) of :math:`\mathbf{K}`.

    Parameters
    ----------
    K : :class:`numpy.ndarray`
        The kernel to normalize.

    Returns
    -------
    :class:`numpy.ndarray`
        The normalized version of the input kernel.
    """
    n = K.shape[0]
    d = np.array([[K[i, i] for i in range(n)]])
    Kn = K / np.sqrt(np.dot(d.T, d))
    return Kn

def swish(x, beta=1.):
    r"""The swish function.

    The swish function is a mathematical function defined as follows:

    :math:`\operatorname{swish}(x) = x \times \operatorname{sigmoid}(\beta x)=\
    \frac{x}{1+e^{-\beta x}}`.

    Parameters
    ----------
    x : :class:`torch.Tensor`
        The input tensor.
    beta : :obj:`float` [optional]
        Multiplicative constant, by default 1. When ``beta``=1, the function becomes equivalent
        to the Sigmoid-weighted Linear Unit function used in reinforcement learning, whereas for
        ``beta``=0, the function turns into the scaled linear function :math:`f(x)=\frac{x}{2}`.
        With ``beta`` approaching infinity, the swish becomes like the ReLU function.
    """
    return x.mul(torch.sigmoid(beta * x))

def log_norm_pdf(x, mu, logvar):
    r"""Lognormal Probability Density Function.

    The lognormal PDF is of the :math:`\mathcal{N}(0,1)` distribution is defined as
    :math:`f(x) = \frac{1}{x} \cdot \frac{1}{\sigma \sqrt{2 \pi}}\
    \exp \left(-\frac{(\ln x-\mu)^{2}}{2 \sigma^{2}}\right)`

    Parameters
    ----------
    x : :class:`torch.Tensor`
        The input tensor.
    mu : :class:`torch.Tensor`
        The mean tensor.
    logvar : :class:`torch.Tensor`
        The tensor representing the logarithm of the variance.
    """
    return -0.5 * (logvar + np.log(2 * np.pi) + (x - mu).pow(2) / logvar.exp())

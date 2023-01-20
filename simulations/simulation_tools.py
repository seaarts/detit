"""
Additional tools for simulation experiments.

These tools are intended to to improve the readability of the simulation
experiments. This module is not considered part of the detit package, but
are included here for the interested reader.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import matthews_corrcoef
from scipy.spatial import distance_matrix


def _logit(score):
    """Logistic function used in logistic_likelihood."""
    return 1 / (1 + tf.exp(-score))


def logit_pred_prob(beta, X):
    """
    Compute predicted probabilites given data and coefficients.

    Returns the predicted probabilities of success
    :math:`p(X, \beta) = \exp\{\beta^T X\} / (1 + \beta^T X)`


    Parameters
    ----------
    beta : [nChains, nDim] tf.tensor
        Coefficient vector(s) of model (supports nChains chains)
    X : [batchSize, nDim] tf.tensor
        Data matrix of with nDim dimensions and batchSize observations.

    Returns
    -------
    p : [nChains, batchSize) tf.tensor
        Predicted probability for each parameter-vector, item pair.
    """
    p = _logit(beta @ tf.transpose(X))
    return p


def logit_log_prob(beta, X, Y):
    """
    Computes loglikelihood given data `X, Y` and coefficients `beta`.

    Parameters
    ----------
    beta : [nChains, nDim] tf.tensor
        Coefficient vector(s). Supports nChains parallel chains.
    X : [batchSize, nDim] tf.tensor
        Data matrix of with nDim dimensions and batchSize observations.
    Y : [batchSize,] tf.tensor
        Vector of binary outcomes.

    Returns
    -------
    log_prob : [nChains,] tf.tensor
        Vector of log-likelihoods, one for each coefficeint vector.
    """
    p = logit_pred_prob(beta, X)
    log_prob = tf.reduce_sum(tf.math.log(p) * Y + tf.math.log(1 - p) * (1 - Y), axis=1)
    return log_prob


def clogit_pred_prob(beta, X):
    """
    Compute predicted probabilities given data and coefficients.

    For each choice set with variables :math:`\{x_1, x_2, \dots, x_J}`
    compute the predicted probability of selecting item :math:`j`
    :math:`p(j\mid X, \beta) = exp(\beta^Tx_j) / \sum^J{k=1}exp(\beta^Tx_k)`.

    Parameters
    ----------
    beta : [nChains, nDims] tensor
        Coefficient vector(s). Supports nChains parallel chains.
    X : [batchSize, nItems, nDims] tensord
        Data tensor of batchSize choices, each choice over nItems items.
        Each item has nDim features. Assumes each choice is over the same
        number of items.

    Returns
    -------
    p : [nChains, nItems, batchSize] tensor
         Tensor of predicted probability for each item in each choice set
         for each parameter vector.
    """
    scores = tf.transpose(tf.matmul(X, beta, transpose_b=True))

    unnorm_probs = tf.exp(scores)
    partition_fn = tf.math.reduce_sum(unnorm_probs, axis=-2)

    p = unnorm_probs / partition_fn[:, tf.newaxis, :]
    return p


def clogit_log_prob(beta, X, Y):
    """
    Computes loglikelihood given data ``X, Y`` and coefficients ``beta``.

    Parameters
    ----------
    beta : [nChains, nDims] tensor
        Coefficient vector(s). Supports `nChains` parallel chains.
    X : [batchSize, nItems, nDims] tensord
        Data tensor of batchSize choices, each choice over nItems items.
        Each item has nDim features. Assumes each choice is over the same
        number of items.
    Y : [batchSize, nItems] tensor
        Collection of binary outcomes. Each row is an length nItems vector;
        should contain only zeros and one 1 indicating the selected item.

    Returns
    -------
    log_prob : [nChains,] tf.tensor
        Vector of log-likelihoods, one for each coefficeint vector.
    """
    p = clogit_pred_prob(beta, X)
    log_probs = tf.math.log(p)

    # sum over all but nChains dimension(s)
    return tf.math.reduce_sum(tf.transpose(Y) * log_probs, axis=[-1, -2])


def dpp_sampler_generic_kernel(K, seed=None):
    """Sample from generic :math:`\\operatorname{DPP}(\\mathbf{K})` with
    potentially non hermitian correlation kernel :math:`\\operatorname{DPP}(\\mathbf{K})`
    based on :math:`LU` factorization procedure.

    Borrowed from DPPy (https://github.com/guilgautier/DPPy/).

    Parameters
    ----------
    K : array_like
        Correlation kernel (potentially non hermitian).

    Returns
    -------
    sample :   array
        A sample :math:`\\mathcal{X}` from :math:`\\operatorname{DPP}(K)` and
        the in-place :math:`LU factorization of :math:`K − I_{\\mathcal{X}^{c}}`
        where :math:`I_{\\mathcal{X}^{c}}` is the diagonal indicator matrix for the
        entries not in the sample :math:`\\mathcal{X}`.
    """

    rng = np.random.default_rng(seed=seed)

    A = K.copy()
    sample = []

    for j in range(len(A)):

        if rng.uniform() < A[j, j]:
            sample.append(j)
        else:
            A[j, j] -= 1

        A[j + 1 :, j] /= A[j, j]
        A[j + 1 :, j + 1 :] -= np.outer(A[j + 1 :, j], A[j, j + 1 :])
        # A[j+1:, j+1:] -=  np.einsum('i,j', A[j+1:, j], A[j, j+1:])

    return sample


def to_negpos(labels):
    """Convert binary labels to -1, 1 labels."""
    return (2 * labels - 1).astype(int)


def summarize_mcc(predictions, y_test):
    """Summarize predictive performance using Matthew's correlation coefficient"""

    nChains = predictions.shape[0]
    matts = []
    for chain in range(nChains):
        y_pred = predictions[chain, :, :].T

        _pred = to_negpos(y_pred).flatten()
        _test = to_negpos(y_test).flatten()

        matts.append(matthews_corrcoef(_test, _pred))

    mean_mcc = np.array(matts).mean()
    stdv_mcc = np.array(matts).std()

    return mean_mcc, stdv_mcc


def sample_square(minx, maxx, size=1, seed=None):
    """
    Samples a given number of points uniformly over a square.

    Parameters
    ----------
    minx : float
        Value for (left, bottom) corner; at (minx, minx).
    maxx : float
        Value for (right, top) corner; at (maxx, maxx)
    size : int, (or tuple of ints)
        Given `size = (s1, ..., sk)` the output is `(s1, ..., sk, 2)`.
    seed : int, (optional)
        Seed value passed to `np.random.default_rng()`.

    Returns
    -------
    points : array
        Array of of points on square.
    """
    rng = np.random.default_rng(seed=seed)

    # append a dimension with 2
    try:
        iter(size)
    except TypeError:
        # not an iterable, assume a single value
        size = [size]
    size = list(size)
    size + [2]

    points = rng.uniform([minx, minx], [maxx, maxx], size=(10, 2))
    return points


def _true_quality(x, beta0=1, beta1=-1):
    """
    Computes and exponential quality based on distance.

    :math:`q(\text{dist})=\frac{1}{1+\exp(-\beta_0+\beta_1\text{dist})}`

    Parameters
    ----------
    x : array-like
        Array of data
    beta0 : float
        A constant; positive constant makes quality higher
    beta1: float
        A coefficient; a negative value mean distance lowers quality.

    Returns
    -------
    qual : array-like
        Array of qualities in [0, 1]
    """
    return 1 / (1 + np.exp(-(beta0 + beta1 * x)))


def sample_matern_iii(marks, dists, radius, labels=None):
    """
    Apply Matern type III thinning on labeled points.

    Metérn type III thinnning is a form of dependent thinning,
    it is a hard-core thinning model, in that points that are
    within the specified `radius` of each other cannot both be
    labeled `1`. Briefly, each point is associated with a real-valued
    mark :math:`m_i`. The point with the largest mark is retained,
    after the point with the smallest mark not within `2*radius` of
    any `1`-labeled points is retained, and so on.

    Parameters
    ----------
    marks : array-like
        A lenth `nPoints` vector of marks.
    dists : array-like
        An (nPoints, nPoints) matrix of pairsiwe distances.
    radius : float
        A non-negative radius of the hard-cores around points.
    labels : array-like, (optional)
        A length `nPoints` vector of binary labels for the points.
        Points already assigned `0`-labels are ignored.

    Returns
    -------
    retained : array-like
       A (nPoints,) array of binary lables; `1`s indicate retention.
    """

    # initialize variables
    nPoints = marks.shape[0]

    # use all ones if labels empty
    if labels is None:
        labels = np.ones(nPoints)

    retained = np.zeros(nPoints).astype(bool)

    indices = np.flip(np.argsort(marks))  # ids of largest marks first

    for i in indices:
        if labels[i]:
            if np.sum((dists[i, :] < 2 * radius) * retained) == 0:
                retained[i] = True

    return retained


def plot_points(points, radius, labels=None, title=None, figsize=(10, 6)):
    """Make plots of 6 trials"""

    fig, ax = plt.subplots(2, 3, figsize=figsize)

    if not labels:
        labels = [np.ones(p.shape[0]) for p in points]

    Ps = [(label == 1) for label in labels]

    for i in range(6):
        X = points[i][:, 0]
        Y = points[i][:, 1]
        # lim = max(np.absolute(X).max()+radius, np.absolute(Y).max()+radius) * 1.1
        lim = 2.2

        # make scatter plot
        sns.scatterplot(
            x=X[Ps[i]], y=Y[Ps[i]], s=25, ax=ax[i // 3, i % 3], color="skyblue"
        )
        sns.scatterplot(
            x=X[~Ps[i]],
            y=Y[~Ps[i]],
            s=25,
            ax=ax[i // 3, i % 3],
            color="grey",
            alpha=0.3,
        )

        # add circle
        for x, y in zip(X[Ps[i]], Y[Ps[i]]):
            circle = plt.Circle(
                xy=(x, y), radius=radius, color="skyblue", zorder=0, alpha=0.5
            )
            ax[i // 3, i % 3].add_patch(circle)
        for x, y in zip(X[~Ps[i]], Y[~Ps[i]]):
            circle = plt.Circle(
                xy=(x, y), radius=radius, color="grey", zorder=0, alpha=0.05
            )
            ax[i // 3, i % 3].add_patch(circle)

        # fromat axes
        ax[i // 3, i % 3].set(title="Trial %d (%d points)" % (i, Ps[i].sum()))
        ax[i // 3, i % 3].set(ylim=(-lim, lim), xlim=(-lim, lim))
    fig.suptitle(title, fontsize=16)
    plt.show()


def simiDataFromPandas(df, cols_simi, trial_id, index_col):
    """
    Extract a list of similarity data arrays from pandas DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Assumed to have a `trial_id`, and `index_col` column, and all the
        variables listed in `vars_simi`.

    cols_simi : list of lists of column names to used for similarity computation.
        Each sublist is one variable group, len(vars_simi) = nGroups
        Pairwise distances treat one variable group as an array.

    trial_id: column name (str) for trial ids in `df`.
       Any column that defines groups (trials) over the rows in `df`.

    index_col:
        A column of unique identifiers for each row.

    Returns
    ----------
    Zs : a list of np.arrays of size [nGroups, nItems, nItesm]
        The number of items nItems may vary between elements,
        if there are variable-sized trials in `df`.
    """

    nGroups = len(cols_simi)

    # get ids grouped by trial
    uplinks = df.groupby(trial_id)[index_col].unique().values
    nTrials = len(uplinks)

    data = []
    for i in range(nGroups):
        pairwiseDists = [df[cols_simi[i]].iloc[ids] for ids in uplinks]
        pairwiseDists = [distance_matrix(X, X) for X in pairwiseDists]
        data.append(pairwiseDists)

    Zs = [np.stack((data[i][j]) for i in range(nGroups)) for j in range(nTrials)]

    return Zs

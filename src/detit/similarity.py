r"""A module for similarity kernel functions.

Similarity kernel functions
---------------------------
A *similarity kernel function* is a function that takes a *set* of :math:`n` items
:math:`B` and associated feature vectors :math:`\mathbf{X}_B \in
\mathbb{R}^{n \times d}`, and generates a *similarity matrix* :math:`S(\mathbf{X})
\in \mathbb{R}^{n \times n}`. This matrix descirbes the similarity :math:`S_{ij}`
for each pair of items :math:`i, j \in B`. The similarity in turn determines the
likelihood of any subset of items :math:`Y \subseteq B` being selected together.
A key feature of similarity kernel functions is that they accept variable-sized
sets :math:`B` and generate symmetric similarity matrices of corresponding size.

Implementation details
----------------------
A key challenge in implementing similarity kernel functions is the variable number
of items per set of items. This is further complicated by vectorizing the
parameter dimension as well as allowing for batching of sets of identical shape,
which is necessary in order to achieve acceptable computational performance.

We assume the data :math:`\mathbf{X}_B` has been expanded into *pairwise*
*distance form*, such that the input is in fact a tensor :math:`\mathbf{Z} = Z
(\mathbf{X}_B)` of shape :math:`(n, n, d)` and where the :math:`(i, j, k)` th
entry is given by

.. math::
    \mathbf{Z}_{i,j,k} = ||\mathbf{x}_{i,k} - \mathbf{x}_{j,k}||

where :math:`\mathbf{x}_i` is the :math:`i` th row of :math:`\mathbf{X}_B`,
corresponding to item :math:`i`, and :math:`||\cdot ||` is a norm of our choosing.
Note that the dimensionality of :math:`\mathbf{Z}` can be samller than that
of :math:`\mathbf{X_B}` in case we use multiple dimensions of the latter to form
one pairwise distance in the former. This can be useful if we have a group of
related variables, such as a collection of dummies, in :math:`\mathbf{X}_B` and
we want entry :math:`\mathbf{Z}_{i,j,k}` to reflect the parisiwe distance between
the dummy *vectors* of two items.

We can now outline the basic inputs of a similarity kernel function. In addition
to the pairwise distance matrix, we allow *batching* of multiple sets :math:`B` of
identical size :math:`n`. We call the data used in similarity kernel functions ``Z``.
This is a kernel of shape ``[batchSize, nItems, nItems, nDims]``, where ``batchSize``
is the number of sets in the batch. A generic similarity kernel funciton also takes
a parameter tensor of dimension ``nDims`` matching the pairsiwe distance data. In
order to conduct inference efficiently we allow parallel chains of parameters for
Bayesian inference. As such ``parameters`` is a tensor of shape ``[nChains, nDims]``,
where ``nChains`` denotes the number of parallel chains.
"""


import tensorflow as tf
from abc import ABC, abstractmethod


class Similarity(ABC):
    """A class of similarity kernel functions."""

    def __init__(self, name):
        self.name = name

    def __repr__(self):
        """Return representative string."""
        return f"Similarity({self.__dict__})"

    @abstractmethod
    def evaluate(self):
        """Evaluate the model on given input.

        All similarity models are assumed to take a data tensor as input.
        """
        pass

    def __call__(self, *args, **kwargs):
        """Call the evaluate function."""
        return self.evaluate(*args, **kwargs)


class RBF(Similarity):
    r"""Class of radial basis function kernels.

    The RBF kernel function takes paiwise distances :math:`\mathbf{Z} \in
    \mathbb{R}^{n \times n \times d}` and a paremeter vector :math:`\mathbf{\ell}`
    called *lengthscales* in :math:`\mathbb{R}^d` and returns an :math:`n`-
    by-:math:`n` matrix :math:`\mathbf{S}` with :math:`(i,j)` th entry

    .. math::
        S_{ij}(\mathbf{Z}, \mathbf{\ell}) =
        \exp\left\{-\sum^d_{k=1}\frac{Z^2_{ijk}}{2\ell^2_k}\right\}

    Note that range of the similarity is in :math:`[0, 1]`.
    """

    def __init__(self):
        super().__init__(name="RBF")

    def evaluate(self, lengthscale, data):
        """
        Evaluate RBF simimlarity kernel function.

        Supports multiple chains ``lengthscale`` as well as bacthed ``data``.

        Parameters
        ----------
        lengthscale : ``tf.tensor`` for shape ``[nChains, nDims]``
            Lengthscales are assumed to be logged, nChains must be 0 or 1 dimensional.

        data : ``[batchSize, nItems, nItems, nDims]``  tensor
            Data tensor of dimension-wise distances. Should be symmetric between items.
            The batchSize must be 0 or 1 dimensional.

        Returns
        -------
        S : ``tf.tensor`` of shape ``[nChains, batchSize, nItems, nItems]``
            A tensor of Similarity kernel matrices.
        """
        lengthscale = 2 * tf.math.exp(lengthscale * 2)  # 2 * ell^2

        # line up dimensions with data
        lengthscale = lengthscale[..., tf.newaxis, tf.newaxis, tf.newaxis, :]
        dists = tf.math.square(data[tf.newaxis, ...]) / lengthscale

        # sum over dimensions
        dist = tf.math.reduce_sum(dists, axis=-1, keepdims=False)

        S = tf.exp(-dist)

        return S

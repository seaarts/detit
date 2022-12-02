"""A module for similarity kernel functions."""

import tensorflow as tf
from abc import ABC, abstractmethod


class Similarity(ABC):
    """A class of similarity kernel functions."""

    def __init__(self, name):
        self.name = name

    @abstractmethod
    def evaluate(self):
        pass


class RBF(Similarity):
    """Class of radial basis function kernels."""

    def __init__(self):
        super().__init__(name="RBF")

    def __repr__(self):
        return self.name

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

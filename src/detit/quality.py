"""A module of Quality kernel functions."""

import tensorflow as tf
from abc import ABC, abstractmethod


class Quality(ABC):
    """Abstract base class of Quality models."""

    def __init__(self, name):
        self.name = name

    def __repr__(self):
        """Return representative string."""
        return f"Quality({self.__dict__})"

    @abstractmethod
    def evaluate(self):
        """Evaluate the quality model on given inputs."""
        pass

    def __call__(self, *args, **kwargs):
        """Evaluate the quality model on given inputs."""
        return self.evaluate(*args, **kwargs)


class Exponential(Quality):
    """An exponential quality model."""

    def __init__(self):
        super().__init__(name="Exponential")

    def evaluate(self, beta, X):
        """
        Evaluate an exponential half-quality model with coefficients `beta`\
        and data `X`.

        Parameters
        ----------
        beta : [nChains, nDims] tensor
            Coefficient vector(s). Supports `nChains` parallel chains.
        X : [batchSize, nItems, nDims] tensor
            Data tensor representing `batchSize` choices over `nItems` items.
            Each item has `nDim` features. Assumes choices in the batch are
            over the same number of items. For different `nItem`-values one
            can use multiple batches.

        Returns
        -------
        Q : [nChains, batchSize, nItems] tensor
            Returns a real-valued quality for each [chain, trial, item]-triple.
        """
        mu = tf.linalg.matmul(beta, X, transpose_b=True)

        Q = tf.math.exp(mu / 2)

        return tf.transpose(Q, perm=[1, 0, 2])

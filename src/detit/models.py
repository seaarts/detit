"""A module for L-ensembles and likelihood functions."""

import tensorflow as tf
from detit.quality import Quality
from detit.similarity import Similarity


class Ensemble:
    """A class of L-ensembles."""

    def __init__(self, quality, similarity):
        """
        Initialize L-ensemble with a quality and similarity model.

        Parameters
        ----------
        quality : detit.quality.Quality instance.
            The quality model to be used in L-ensemble.

        similarity : detit.similarity.Similarity instance.
            The similarity model to be used in L-ensemble.

        Raises
        ------
        ValueError
            If quality is not Quality or similarity not Similarity.
        """
        if not isinstance(quality, Quality):
            raise ValueError("The `quality` argument must be a Quality istance.")
        if not isinstance(similarity, Similarity):
            raise ValueError(
                "The `similarity` argument must be a\
                 Similarity instance."
            )
        self.qual = quality
        self.simi = similarity

    def evaluate(self, paramsQual, paramsSimi, X, Z):
        """
        Evaluate L-ensemble with sub-model arguments.

        The inputs must match those of the quality, and similarity models,
        respectively.
        """
        Q = self.qual(paramsQual, X)
        S = self.simi(paramsSimi, Z)

        return Q[:, :, tf.newaxis, :] * S * Q[:, :, :, tf.newaxis]

    def __call__(self, paramsQual, paramsSimi, X, Z):
        """Call evaluate function."""
        return self.evaluate(paramsQual, paramsSimi, X, Z)

    def log_det(self, beta, lengthscale, element, eye=False):
        """
        Evaluate the log-determinant of the L-ensemble.

        Parameters
        ----------
        params_qual : [nChains, nDims] tensor
            Coefficient vector(s) for the quality model.
        lengthscale : [nChains, nDims] tensor
            Lengthscale vector(s). Supports `nChains` parallel chains.
        element : list, from tf.dataset
            The dataset is assumet to have 3 components:
            element[0] : int, number of items
            element[1] : tensor, quality data
            element[2] : tensor, similarity data
        eye : Bool, optional
            Optional boolean for whether to add an identity tensor to\
            the L-ensemble prior to taking the log det. Used for DPP\
            log-likelihood evaluations.

        Returns
        -------
        log_det : tf.double
            The log-determinant of L-ensemble evaluated on `element`.
        """
        pass

        # READ about tf parameter and or data objects

        # L = self(beta, lengthscale, element[1], element[2])

        # if eye:
        # chains, N, n =  L.shape[:3]
        # L += tf.eye(num_rows=n, batch_shape=[chains, N], dtype=L.dtype)

        # return tf.math.reduce_sum(tf.linalg.logdet(L), axis=1)

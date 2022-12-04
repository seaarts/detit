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

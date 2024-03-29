"""A module for L-ensembles and likelihood functions."""

import tensorflow as tf
import numpy as np
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

    def logdet(self, beta, lengthscale, element, eye=False):
        """
        Evaluate the log-determinant of the L-ensemble.

        Parameters
        ----------
        beta : [nChains, nDims] tensor
            Parameters vector(s) for the quality model.
        lengthscale : [nChains, nDims] tensor
            Lengthscale vector(s) for similarity model.
        element : list, from tf.dataset object.
            The dataset is assumed to have 3 components:
            element[0] : int, number of items
            element[1] : tensor, quality data
            element[2] : tensor, similarity data
            The tensors are expected to have a batch dimension.
        eye : Bool, optional
            Optional boolean for whether to add an identity tensor to\
            the L-ensemble prior to taking the log det. Used for DPP\
            log-likelihood evaluations.

        Returns
        -------
        log_det : tf.double
            The log-determinant of L-ensemble evaluated on `element`.

        Notes
        -----
        We need to pass the number of items (``element[0]``) to the function
        explicitly, as ``tensorflow`` will not always explicitly form the tensors,
        so their shape mid-run may evaluate to ``None``. This would raise an error.
        To remedy this size information about the tensors is passed explicitly.
        """
        L = self(beta, lengthscale, element[1], element[2])

        if eye:
            chains, N, n = L.shape[:3]
            L += tf.eye(num_rows=n, batch_shape=[chains, N], dtype=L.dtype)

        return tf.math.reduce_sum(tf.linalg.logdet(L), axis=1)

    def log_likelihood(self, beta, lengthscale, dataset_succ, dataset_full):
        """
        Evaluate log-likelihood of DPP model.

        Parameters
        ----------
        beta : tf.tensor
            Parameter tensor for quality model.
        lengthsscale : tf.tensor
            Parametere rensor for similarity model.
        dataset_succ : tf.Dataset
            Dataset of 1-labeled observations.
            Should contain (nItems, X, Z).
        dataset_full : tf.Dataset
            Dataset of all observations.
        """
        loglik = tf.constant(np.zeros(beta.shape[0]))

        # add denominator components
        for element in dataset_full:
            loglik -= self.logdet(beta, lengthscale, element, eye=True)

        # add numerator components
        if not dataset_succ:
            return loglik
        for element in dataset_succ:
            loglik += self.logdet(beta, lengthscale, element, eye=False)

        return loglik

    def likelihoodClosure(self, dataset, nDimsQual, prior=None):
        """
        Returns a functional closure of the log-likelihood.

        This creates a simple *functional closure* of the log-likelihood. The
        main purpose is to attach a dataset to the function, and to make the
        input accept a single parameter tensor, rather than two separate ones
        for the quality and similarity models, respectively. The closure can
        then be passed to e.g. a minimizer or Bayesian inference function.

        Parameters
        ----------
        dataset : detit.data.Dataset
            A batched dataset of successes and full observations.
        nDimsQual : int
            The parameter dimension of the quality model.
        prior : optional, tensorflow.probablity.distribution-object
            A prior tfp.distribution with ``event_shape`` equal to ``nDimsQual +
            nDimsSimi``.

        Returns
        -------
        log_prob : function
            A function that takes parameters and returns a log-likelihood at
            of the given parameters and dataset.
        """
        if not prior:

            def _log_prob(params):
                loglik = self.log_likelihood(
                    params[:, :nDimsQual],
                    params[:, nDimsQual:],
                    dataset.succ,
                    dataset.full,
                )
                return loglik

        else:

            def _log_prob(params):
                loglik = self.log_likelihood(
                    params[:, :nDimsQual],
                    params[:, nDimsQual:],
                    dataset.succ,
                    dataset.full,
                )
                return loglik + prior.log_prob(params)

        return _log_prob

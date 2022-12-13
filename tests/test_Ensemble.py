import pytest
import numpy as np
import tensorflow as tf
import detit.data as dtd
from detit.models import Ensemble
from detit.quality import Exponential
from detit.similarity import RBF


@pytest.fixture()
def makeEnsemble():
    qual = Exponential()
    simi = RBF()
    return Ensemble(qual, simi)


class TestEnsemble:
    def test_props(self):
        """Ensure correct models can be successfully used."""
        qual = Exponential()
        simi = RBF()
        ell = Ensemble(qual, simi)
        assert (ell.qual, ell.simi) == (qual, simi)

    @pytest.mark.parametrize(
        "qual, simi",
        [
            (None, None),
            (1, 2),
            (Exponential(), None),
            (None, RBF()),
            (RBF(), Exponential()),  # positions flipped
        ],
    )
    def test_raises(self, qual, simi):
        """Verify value error raised if inputs are imporper."""
        with pytest.raises(ValueError):
            Ensemble(qual, simi)

    @pytest.mark.parametrize(
        "pqual, psimi, xShape, zShape, expected",
        [
            ((1, 2), (1, 1), (1, 3, 2), (1, 3, 3, 1), (1, 1, 3, 3)),
            ((5, 2), (5, 1), (1, 3, 2), (1, 3, 3, 1), (5, 1, 3, 3)),
            ((5, 2), (5, 1), (7, 3, 2), (7, 3, 3, 1), (5, 7, 3, 3)),
            ((2, 2), (2, 1), (7, 9, 2), (7, 9, 9, 1), (2, 7, 9, 9)),
        ],
    )
    def test_evaluate(self, pqual, psimi, xShape, zShape, expected, makeEnsemble):
        """Check evaluations using simple binary inputs."""
        ell = makeEnsemble

        paramsQual = tf.zeros(shape=pqual, dtype=tf.double)
        paramsSimi = tf.ones(shape=psimi, dtype=tf.double)
        X = tf.ones(shape=xShape, dtype=tf.double)
        Z = tf.ones(shape=zShape, dtype=tf.double)

        L = ell.evaluate(paramsQual, paramsSimi, X, Z)

        assert L.shape == expected

    @pytest.mark.parametrize(
        "pqual, psimi, xShape, zShape, expected",
        [
            ((1, 2), (1, 1), (1, 3, 2), (1, 3, 3, 1), (1, 1, 3, 3)),
            ((5, 2), (5, 1), (1, 3, 2), (1, 3, 3, 1), (5, 1, 3, 3)),
            ((5, 2), (5, 1), (7, 3, 2), (7, 3, 3, 1), (5, 7, 3, 3)),
            ((2, 2), (2, 1), (7, 9, 2), (7, 9, 9, 1), (2, 7, 9, 9)),
        ],
    )
    def test_call(self, pqual, psimi, xShape, zShape, expected, makeEnsemble):
        """Check evaluations using simple binary inputs."""
        ell = makeEnsemble

        paramsQual = tf.zeros(shape=pqual, dtype=tf.double)
        paramsSimi = tf.ones(shape=psimi, dtype=tf.double)
        X = tf.ones(shape=xShape, dtype=tf.double)
        Z = tf.ones(shape=zShape, dtype=tf.double)

        L = ell(paramsQual, paramsSimi, X, Z)

        assert L.shape == expected

    @pytest.mark.parametrize(
        "shapeBeta, shapeLs, dataShape, eye, expected",
        [
            ((5, 2), (5, 1), (3, (1, 3, 2), (1, 3, 3, 1)), False, (5)),
            ((1, 2), (1, 1), (3, (1, 3, 2), (1, 3, 3, 1)), False, (1)),
            ((2, 9), (2, 1), (3, (1, 8, 9), (1, 8, 8, 1)), False, (2)),
            ((5, 2), (5, 1), (3, (1, 3, 2), (1, 3, 3, 1)), True, (5)),
            ((1, 2), (1, 1), (3, (1, 3, 2), (1, 3, 3, 1)), True, (1)),
            ((2, 9), (2, 1), (3, (1, 8, 9), (1, 8, 8, 1)), True, (2)),
        ],
    )
    def test_logdet(self, shapeBeta, shapeLs, dataShape, eye, expected, makeEnsemble):
        ensemble = makeEnsemble
        paramsQual = tf.zeros(shape=shapeBeta, dtype=tf.double)
        paramsSimi = tf.ones(shape=shapeLs, dtype=tf.double)
        X = tf.ones(shape=dataShape[1], dtype=tf.double)
        Z = tf.ones(shape=dataShape[2], dtype=tf.double)

        logDets = ensemble.logdet(paramsQual, paramsSimi, (dataShape[0], X, Z), eye)

        assert logDets.shape == expected

    @pytest.mark.parametrize(
        "beta, ell, labels, Xs, Zs, expected",
        [
            (
                np.ones(shape=(1, 1)),
                np.ones(shape=(1, 1)),
                [np.array([True])],
                [np.ones(shape=(1, 1))],
                [np.zeros(shape=(1, 1, 1))],
                1 - np.log(1 + np.e),
            ),
            (
                np.ones(shape=(1, 1)),
                np.ones(shape=(1, 1)),
                [np.array([False])],
                [np.ones(shape=(1, 1))],
                [np.zeros(shape=(1, 1, 1))],
                np.log(1) - np.log(1 + np.e),
            ),
            (
                np.ones(shape=(1, 1)),
                np.ones(shape=(1, 1)),
                [np.array([True]), np.array([True])],
                [np.ones(shape=(1, 1)), np.ones(shape=(1, 1))],
                [np.zeros(shape=(1, 1, 1)), np.zeros(shape=(1, 1, 1))],
                2 * (1 - np.log(1 + np.e)),
            ),
        ],
    )
    def test_log_likelihood(self, beta, ell, labels, Xs, Zs, expected, makeEnsemble):
        """Form ensemble and detit.Dataset and evaluate.

        Notes
        -----
        We use identity matrices for Zs in order to keep calculations
        of expected log-likelihoods simple. We can vary the parameters
        beta and the number of items, dimensions, and observations to
        induce variation in the testing examples.
        """

        ensemble = makeEnsemble

        ds = dtd.Dataset.fromNumpy(labels, Xs, Zs)

        loglik = ensemble.log_likelihood(beta, ell, ds.succ, ds.full)

        assert np.allclose(loglik.numpy()[0], expected)

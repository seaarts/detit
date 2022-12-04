import pytest
import numpy as np
import tensorflow as tf
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
            ell = Ensemble(qual, simi)

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

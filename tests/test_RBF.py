import pytest
import tensorflow as tf
import numpy as np
import detit.similarity as simi


class TestRBF:
    def test_props(self):
        rbf = simi.RBF()
        assert rbf.name == "RBF"

    def test_repr(self):
        rbf = simi.RBF()
        assert rbf.__repr__() == rbf.name

    @pytest.mark.parametrize(
        "lsShape, datShape, expShape",
        [
            ((1), (3, 3, 1), (1, 3, 3)),
            ((1), (1, 3, 3, 1), (1, 1, 3, 3)),
            ((2, 1), (3, 3, 1), (2, 1, 3, 3)),
            ((5, 2), (4, 3, 3, 2), (5, 4, 3, 3)),
        ],
    )
    def test_evaluate_shape(self, lsShape, datShape, expShape):
        ls = tf.zeros(shape=lsShape, dtype=tf.double)
        dat = tf.ones(shape=datShape, dtype=tf.double)
        rbf = simi.RBF()
        assert rbf.evaluate(ls, dat).shape == expShape

    @pytest.mark.parametrize(
        "lsShape, datShape, expShape",
        [
            ((1), (3, 3, 1), (1, 3, 3)),
            ((1), (1, 3, 3, 1), (1, 1, 3, 3)),
            ((2, 1), (3, 3, 1), (2, 1, 3, 3)),
            ((5, 2), (4, 3, 3, 2), (5, 4, 3, 3)),
        ],
    )
    def test_evaluate_val(self, lsShape, datShape, expShape):
        """Zeros and ones only so that we expect exp(-nDims/2)"""
        rbf = simi.RBF()
        ls = tf.zeros(shape=lsShape, dtype=tf.double)
        dat = tf.ones(shape=datShape, dtype=tf.double)
        out = tf.ones(shape=expShape, dtype=tf.double) * (np.e ** (-datShape[-1] / 2))

        assert np.allclose(rbf.evaluate(ls, dat).numpy(), out.numpy())

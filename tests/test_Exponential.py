import pytest
import tensorflow as tf
import numpy as np
from detit.quality import Exponential


class TestExponential:
    def test_name(self):
        qual = Exponential()
        assert qual.name == "Exponential"

    def test_repr(self):
        qual = Exponential()
        assert qual.__repr__() == "Quality({'name': 'Exponential'})"

    @pytest.mark.parametrize(
        "beta, X, expected",
        [
            ([[0, 1]], [[[0, 2]]], [np.e]),
            ([[1, 1]], [[[1, 1]]], [np.e]),
            ([[1, 1, 1, 1]], [[[0.5, 0.5, 0.5, 0.5]]], [np.e]),
            (np.ones((1, 5)), (2 / 5) * np.ones((1, 1, 5)), [np.e]),
            (np.ones((1, 5)), (2 / 5) * np.ones((1, 10, 5)), np.e * np.ones((1, 10))),
            (
                np.ones((2, 5)),
                (2 / 5) * np.ones((1, 10, 5)),
                np.e * np.ones((2, 1, 10)),
            ),
            (
                np.ones((2, 5)),
                (2 / 5) * np.ones((14, 10, 5)),
                np.e * np.ones((2, 14, 10)),
            ),
        ],
    )
    def test_evaluate(self, beta, X, expected):
        qual = Exponential()

        beta = tf.constant(beta, dtype=tf.double)
        X = tf.constant(X, dtype=tf.double)

        assert np.allclose(qual.evaluate(beta, X).numpy(), np.array(expected))

    @pytest.mark.parametrize(
        "beta, X, expected",
        [
            ([[0, 1]], [[[0, 2]]], [np.e]),
            ([[1, 1]], [[[1, 1]]], [np.e]),
            ([[1, 1, 1, 1]], [[[0.5, 0.5, 0.5, 0.5]]], [np.e]),
            (np.ones((1, 5)), (2 / 5) * np.ones((1, 1, 5)), [np.e]),
            (np.ones((1, 5)), (2 / 5) * np.ones((1, 10, 5)), np.e * np.ones((1, 10))),
            (
                np.ones((2, 5)),
                (2 / 5) * np.ones((1, 10, 5)),
                np.e * np.ones((2, 1, 10)),
            ),
            (
                np.ones((2, 5)),
                (2 / 5) * np.ones((14, 10, 5)),
                np.e * np.ones((2, 14, 10)),
            ),
        ],
    )
    def test_call(self, beta, X, expected):
        qual = Exponential()

        beta = tf.constant(beta, dtype=tf.double)
        X = tf.constant(X, dtype=tf.double)

        assert np.allclose(qual(beta, X).numpy(), np.array(expected))

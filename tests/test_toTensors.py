import pytest
import tensorflow as tf
from detit.data import toTensors


@pytest.mark.parametrize(
    "data, dtype, expected",
    [
        ([[0, 1]], tf.double, [tf.constant([0, 1], dtype=tf.double)]),
        ([[1, 2]], tf.int64, [tf.constant([1, 2], dtype=tf.int64)]),
        (
            [[1], [2]],
            tf.double,
            [tf.constant([1], dtype=tf.double), tf.constant([2], dtype=tf.double)],
        ),
        (
            [[1, 2], [2, 3]],
            tf.double,
            [
                tf.constant([1, 2], dtype=tf.double),
                tf.constant([2, 3], dtype=tf.double),
            ],
        ),
    ],
)
def test_toTensors(data, dtype, expected):
    tensors = toTensors(data, dtype)
    assert all(tf.math.reduce_all(tf.equal(t, e)) for t, e in zip(tensors, expected))

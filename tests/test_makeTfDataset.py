import tensorflow as tf
import pytest
import numpy as np

from detit.data import makeTfDataset


@pytest.mark.parametrize(
    "shapeX, shapeZ, expect",
    [
        ([4, 3], [4, 4, 6], (3, 6)),
        ([9, 2], [9, 9, 6], (2, 6)),
        ([1, 2], [1, 1, 7], (2, 7)),
    ],
)
def test_makeTfDataset(shapeX, shapeZ, expect):

    Xs = [tf.ones(shape=shapeX, dtype=tf.double)]
    Zs = [tf.ones(shape=shapeZ, dtype=tf.double)]

    dataset = makeTfDataset(Xs, Zs)

    specs = [s.shape for s in dataset.element_spec]

    assert (specs[1][-1], specs[2][-1]) == expect


@pytest.mark.parametrize(
    "Xs, Zs",
    [
        ([0, 0], [0, 0, 0]),  # non-tensor
        ([np.ones((3, 2))], [tf.ones((3, 3, 2)), tf.ones((3, 3, 2))]),  # NumPy in Xs
        ([tf.ones((3, 2))], [tf.ones((3, 3, 2)), np.ones((3, 3, 2))]),  # NumPy in Zs
        (
            [tf.ones((3, 2))],
            [tf.ones((3, 3, 2)), tf.ones((3, 3, 2))],
        ),  # nTrial mismatch
        (
            [tf.ones((3, 2)), tf.ones((3, 2))],
            [tf.ones((3, 3, 2)), tf.ones((3, 2, 2))],
        ),  # nItems mismatch in one Z in Zs
        (
            [tf.ones((3, 2)), tf.ones((3, 2))],
            [tf.ones((4, 4, 2)), tf.ones((3, 3, 2))],
        ),  # nItems mismatch between Zs
        (
            [tf.ones((3, 4)), tf.ones((3, 5))],
            [tf.ones((3, 3, 4)), tf.ones((3, 3, 4))],
        ),  # nDims mismatch in Xs
        (
            [tf.ones((3, 4)), tf.ones((3, 4))],
            [tf.ones((3, 3, 4)), tf.ones((3, 3, 5))],
        ),  # nDims mismatch in Zs
    ],
)
def test_makeTfDatasetBad(Xs, Zs):
    with pytest.raises(ValueError):
        makeTfDataset(Xs, Zs)

import pytest
import numpy as np
from detit.data import _isDataset
import tensorflow as tf


@pytest.mark.parametrize("data", [[], [1, 2, 3], [[1, 2], [3, 4], [4, 5]]])
def test_isDataset(data):
    dataset = tf.data.Dataset.from_tensor_slices(data)
    dataset = _isDataset(dataset)
    assert dataset


@pytest.mark.parametrize(
    "data",
    [
        [0],
        [1, 2, 3],
        np.ones(10),
        {"data": [0, 1, 2]},
    ],
)
def test_isDataset_bad(data):
    with pytest.raises(ValueError):
        _isDataset(data)

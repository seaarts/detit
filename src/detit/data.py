"""A module for organizing the data needed learning determinantal choice models.

Subset choice modeling, and determinantal choice modeling in particular,
introduces some challenges with respect to storing data. The main source
of difficulty is that the data are not uniformly shaped across
observations. Some choices can be made over sets of 2 items, others over
sets of 100 items. As a consequence, we cannot store all observations in
one array, as this would necessarily make the array ragged. Moreover,
we want to seamlessly subset collections of items based on whether or not
they were selected. To manage all this, we indroduce the ``Dataset`` object.
"""

import tensorflow as tf
import numpy as np


def _isDataset(data):
    """Check if input is a tf.data.Dataset."""
    if not data:
        return None
    if isinstance(data, tf.data.Dataset):
        return data
    else:
        raise ValueError("Input must be a tf.data.Dataset.")


def toTensors(data, dtype=None):
    """Convert arrays in list of data into tf.Tensors.

    Paramaters
    ----------
    data : list of array-like
        List of input data to be made into tensors.
        Individual entries should *not* be ragged.

    dtype : tf.dtypes.DType
        A datatype valid for tensorflow.

    See Also
    --------
    Tensorflow documentation for `tf.dtypes\
        <https://www.tensorflow.org/api_docs/python/tf/dtypes>`_
    """
    return [tf.constant(d, dtype=dtype) for d in data]


def makeTfDataset(Xs, Zs):
    """Make a tensorflow.data.Dataset from lists of tensors.

    Verify that the input data is *valid* and make a tf.data.Dataset.
    For the data to be valid there must be an equal number of quality
    data arrays ``Xs`` as similarity arrays ``Zs``. For one entry ``i``
    the number of items in ``Xs[i]`` and ``Zs[i]`` must match. Each entry
    in ``Xs`` must have the same dimensionality ``dimsQual``. Similarly,
    each entry in ``Zs`` must have the same dimensionality ``dimsSimi``.
    The two dimensionalities can differ.

    Parameters
    ----------
    Xs : list of quality data
        Each entry i should be of shape [nItems(i), dimsQual]
        The number of items can vary but the dimnensions must be uniform.

    Zs : list of similarity data
        Each entry i should be of shape [nItems(i), nItems(i), dimsSimi].
        The number of items can vary by the dimensions must be uniform.
    """
    if not all([isinstance(X, tf.Tensor) for X in Xs]):
        raise ValueError("Not all entries in Xs are tf.Tensors.")

    if not all([isinstance(Z, tf.Tensor) for Z in Zs]):
        raise ValueError("Not all entries in Zs are tf.Tensors.")

    if not len(Xs) == len(Zs):
        raise ValueError("Lengths of Xs and Zs not equal.")

    nItems = [x.shape[0] for x in Xs]
    nItemsZ = [z.shape[0] for z in Zs]
    if not all([n == k for n, k in zip(nItems, nItemsZ)]):
        raise ValueError("Mismatch in number of items between entry in Xs and Zs.")
    nItemsZ2 = [z.shape[1] for z in Zs]
    if not all([n == k for n, k in zip(nItemsZ, nItemsZ2)]):
        raise ValueError("Some Z in Zs has Z.shape[0] != Z.shape[1].")

    # ensure dims are uniform
    if not all(x.shape[-1] == Xs[0].shape[-1] for x in Xs):
        raise ValueError("Some Xs have different dimensionality.")
    if not all(z.shape[-1] == Zs[0].shape[-1] for z in Zs):
        raise ValueError("Some Zs have different dimesnionality.")

    dimsQual = Xs[0].shape[-1]  # dims in quality model
    dimsSimi = Zs[0].shape[-1]  # dims in similarity model

    ds_n = tf.data.Dataset.from_tensor_slices(nItems)
    ds_X = tf.data.Dataset.from_generator(
        lambda: Xs,
        output_signature=tf.TensorSpec(shape=[None, dimsQual], dtype=tf.double),
    )
    ds_Z = tf.data.Dataset.from_generator(
        lambda: Zs,
        output_signature=tf.TensorSpec(shape=[None, None, dimsSimi], dtype=tf.double),
    )
    return ds_n.zip((ds_n, ds_X, ds_Z))


class Dataset:
    """A class of datasets for determinantal choice modeling.

    .. note::
        The ``Dataset`` object maintains two *batched* ``tensorflow.data.Datasets``.
        This means that tensors have shape starting with ``[batchSize, ...]``. This
        is the preferred input to the L-ensemble implementation.

    """

    def _n_obs(self, ele0, ele1, ele2):
        return ele0

    def __init__(self, data_succ, data_full, maxBucketSize=1000):
        """Initialize dataset from inputs."""
        succ = _isDataset(data_succ)
        full = _isDataset(data_full)

        nMax = max([self._n_obs(*ele).numpy() for ele in full])

        batched_full = full.bucket_by_sequence_length(
            element_length_func=self._n_obs,
            bucket_boundaries=np.arange(1, nMax + 1),
            bucket_batch_sizes=np.repeat(maxBucketSize, nMax + 1),
        )

        if succ:
            batched_succ = succ.bucket_by_sequence_length(
                element_length_func=self._n_obs,
                bucket_boundaries=np.arange(1, nMax + 1),
                bucket_batch_sizes=np.repeat(maxBucketSize, nMax + 1),
            )
        else:
            batched_succ = succ

        self.succ = batched_succ
        self.full = batched_full

    @classmethod
    def fromNumpy(cls, labels, Xs, Zs, dtype=tf.double):
        """Make a Dataset-instance from lists of numpy arrays.

        This function generates two tensorflow Dataset-objects and stores them
        as a detit Dataset-object. The first consists of a subset of data
        corresponding only to items with postive (``True``) labels. If a trial
        has no positive labels it is not included. The second dataset consits of
        the full data. The data being subsetted is denoted ``Xs`` and ``Zs``,
        represending quality and similarity data, respectively.

        Parameters
        ----------
        labels : list of numpy.arrays
            List of (nItems, ) boolean numpy.arrays whith True indicating success.
        Xs : list of numpy.arrays
            List of (nItems, dimsQual) numpy.arrays of quality-model data.
        Zs : list of numpy.arrays
            List of (nItems, nItems, dimsSimi) symmetric numpy.arrays of
            similarity-model data.
        dtype : dtatype, optional
            tf.datatype, defaults to tf.double.

        Returns
        -------
        dataset : detit.data.Dataset
            A dataset object for likelihood model.
        """
        # get trials with positive number of successes
        succs = [i for i, y in enumerate(labels) if y.sum() > 0]
        Xs_succ = [Xs[i][labels[i]] for i in succs]
        Zs_succ = [Zs[i][labels[i]][:, labels[i], :] for i in succs]

        # make tf.tensors
        Xs_succ = toTensors(Xs_succ, dtype=dtype)
        Zs_succ = toTensors(Zs_succ, dtype=dtype)
        Xs = toTensors(Xs, dtype=dtype)
        Zs = toTensors(Zs, dtype=dtype)

        if Xs_succ and Zs_succ:
            dataset_succ = makeTfDataset(Xs_succ, Zs_succ)
        else:
            dataset_succ = None
        dataset_full = makeTfDataset(Xs, Zs)

        return cls(dataset_succ, dataset_full)

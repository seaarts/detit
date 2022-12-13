import pytest
import numpy as np
import tensorflow as tf
from detit.data import Dataset, makeTfDataset, toTensors


class TestDataset:
    def test_init(self):
        succ = makeTfDataset([tf.ones((3, 3))], [tf.ones((3, 3, 2))])
        full = makeTfDataset([tf.ones((9, 3))], [tf.ones((9, 9, 2))])
        # dss = makeTfDataset(tf.constant(3), [tf.ones((3, 3))], [tf.ones((3, 3, 2))])
        # dsf = makeTfDataset([tf.constant(9), tf.ones((9, 3))], [tf.ones((9, 9, 2))])

        def _nobs(e1, *args):
            return e1

        nMax = max([_nobs(*ele).numpy() for ele in full])

        # batch trials by nr of observations
        bdss = succ.bucket_by_sequence_length(
            element_length_func=_nobs,
            bucket_boundaries=np.arange(1, nMax + 1),
            bucket_batch_sizes=np.repeat(1000, nMax + 1),
        )

        # batch trials by nr of observations
        bdsf = full.bucket_by_sequence_length(
            element_length_func=_nobs,
            bucket_boundaries=np.arange(1, nMax + 1),
            bucket_batch_sizes=np.repeat(1000, nMax + 1),
        )

        ds = Dataset(succ, full)

        succ_same = tf.data.DatasetSpec.from_value(
            ds.succ
        ) == tf.data.DatasetSpec.from_value(bdss)
        full_same = tf.data.DatasetSpec.from_value(
            ds.full
        ) == tf.data.DatasetSpec.from_value(bdsf)

        assert succ_same and full_same

    @pytest.mark.parametrize(
        "labels, Xs, Zs, Xs_succ, Zs_succ",
        [
            (
                [np.array([True, True])],
                [np.ones((2, 3))],
                [np.ones((2, 2, 3))],
                [tf.ones((2, 3))],
                [tf.ones((2, 2, 3))],
            ),
            (
                [np.array([True, True, False])],
                [np.ones((3, 5))],
                [np.ones((3, 3, 5))],
                [tf.ones((2, 5))],
                [tf.ones((2, 2, 5))],
            ),
            (
                [np.array([False, False, False])],
                [np.ones((3, 5))],
                [np.ones((3, 3, 5))],
                [],
                [],
            ),
        ],
    )
    def test_fromNumpy(self, labels, Xs, Zs, Xs_succ, Zs_succ):
        dataset = Dataset.fromNumpy(labels, Xs, Zs)

        if Xs_succ and Zs_succ:
            succ = makeTfDataset(Xs_succ, Zs_succ)
        else:
            succ = None
        full = makeTfDataset(toTensors(Xs), toTensors(Zs))

        # assert nr elements match across entries
        if succ:
            succGood = all([e1[0] == e2[0] for e1, e2 in zip(dataset.succ, succ)])
        else:
            succGood = True
        fullGood = all([e1[0] == e2[0] for e1, e2 in zip(dataset.full, full)])

        assert succGood and fullGood

"""Unit tests for the rectorch.samplers module
"""
import os
import sys
import pytest
import numpy as np
import pandas as pd
import torch
import scipy
sys.path.insert(0, os.path.abspath('..'))
import tempfile, json

from rectorch import set_seed
from rectorch.data import Dataset, NCRDataProcessing
from rectorch.models.nn.svae import SVAE_Sampler
from rectorch.models.nn.cfgan import CFGAN_Sampler
from rectorch.models.nn.ncr import NCR_Sampler
from rectorch.samplers import Sampler, DataSampler, DictDummySampler,\
    ArrayDummySampler, TensorDummySampler, SparseDummySampler

from rectorch.models.nn.cvae import EmptyConditionedDataSampler, ConditionedDataSampler

def test_Sampler():
    """Test the Sampler class
    """
    sampler = Sampler(None, None)
    sampler.train()
    assert sampler.mode == "train"
    sampler.valid()
    assert sampler.mode == "valid"
    sampler.test()
    assert sampler.mode == "test"

    with pytest.raises(NotImplementedError):
        len(sampler)

    with pytest.raises(NotImplementedError):
        for _ in sampler:
            pass

    values = [1., 1., 1., 1.]
    rows = [0, 0, 1, 1]
    cols = [0, 1, 1, 2]
    df_tr = pd.DataFrame(list(zip(rows, cols, values)), columns=['uid', 'iid', 'rating'])
    df_te_tr = pd.DataFrame([(0, 0, 1.)], columns=['uid', 'iid', 'rating'])
    df_te_te = pd.DataFrame([(0, 1, 1.)], columns=['uid', 'iid', 'rating'])
    uids = {0:0, 1:1}
    iids = {0:0, 1:1, 2:2}
    data = Dataset(df_tr, None, (df_te_tr, df_te_te), uids, iids)

    ads = Sampler.build(data, **{"name":"ArrayDummySampler", "mode":"train", "batch_size":2})
    ads.train()
    for tr, te in ads:
        assert isinstance(tr, tuple)
        assert te is None
        assert tr[0] == [0, 1]
        assert np.all(tr[1] == [[1, 1, 0], [0, 1, 1]])

    ads.test()
    for tr, te in ads:
        assert isinstance(tr, tuple)
        assert tr[0] == [0]
        assert np.all(tr[1] == [[1, 0, 0]])
        assert np.all(te == [[0, 1, 0]])

def test_DummySampler():
    """Test the hierarchy of dummy samplers
    """
    values = [1., 1., 1., 1.]
    rows = [0, 0, 1, 1]
    cols = [0, 1, 1, 2]
    df_tr = pd.DataFrame(list(zip(rows, cols, values)), columns=['uid', 'iid', 'rating'])
    df_te_tr = pd.DataFrame([(0, 0, 1.)], columns=['uid', 'iid', 'rating'])
    df_te_te = pd.DataFrame([(0, 1, 1.)], columns=['uid', 'iid', 'rating'])
    uids = {0:0, 1:1}
    iids = {0:0, 1:1, 2:2}
    data = Dataset(df_tr, None, (df_te_tr, df_te_te), uids, iids)

    dds = DictDummySampler(data, shuffle=False)
    assert isinstance(dds.data_tr, dict)
    ads = ArrayDummySampler(data, shuffle=False)
    assert isinstance(ads.data_tr, np.ndarray)
    tds = TensorDummySampler(data, shuffle=False)
    assert isinstance(tds.data_tr, (torch.FloatTensor, torch.sparse.FloatTensor))
    sds = SparseDummySampler(data, shuffle=False)
    assert isinstance(sds.data_tr, scipy.sparse.csr_matrix)

    assert isinstance(dds.data_te[0], dict)
    assert isinstance(ads.data_te[0], np.ndarray)
    assert isinstance(tds.data_te[0], (torch.FloatTensor, torch.sparse.FloatTensor))
    assert isinstance(sds.data_te[0], scipy.sparse.csr_matrix)

    dds.train()
    for tr in dds:
        assert isinstance(tr, tuple)
        assert tr[0][0] == [0, 1]
        assert tr[0][1] == [[0, 1], [1, 2]]

    dds.test()
    for tr, te in dds:
        assert isinstance(tr, tuple)
        assert tr[0] == [0]
        assert te[0] == [1]

    ads.train()
    for tr, te in ads:
        assert isinstance(tr, tuple)
        assert te is None
        assert tr[0] == [0, 1]
        assert np.all(tr[1] == [[1, 1, 0], [0, 1, 1]])

    ads.test()
    for tr, te in ads:
        assert isinstance(tr, tuple)
        assert tr[0] == [0]
        assert np.all(tr[1] == [[1, 0, 0]])
        assert np.all(te == [[0, 1, 0]])

    tds.train()
    for tr, te in tds:
        assert isinstance(tr, tuple)
        assert te is None
        assert tr[0] == [0, 1]
        assert torch.all(tr[1] == torch.FloatTensor([[1, 1, 0], [0, 1, 1]]))

    tds.test()
    for tr, te in tds:
        assert isinstance(tr, tuple)
        assert tr[0] == [0]
        assert torch.all(te == torch.FloatTensor([[0, 1, 0]]))
        assert torch.all(tr[1] == torch.FloatTensor([[1, 0, 0]]))

    sds.train()
    for tr, te in sds:
        assert isinstance(tr, tuple)
        assert te is None
        assert tr[0] == [0, 1]
        assert np.all(tr[1].toarray() == [[1, 1, 0], [0, 1, 1]])

    sds.test()
    for tr, te in sds:
        assert isinstance(tr, tuple)
        assert tr[0] == [0]
        assert np.all(tr[1].toarray() == [[1, 0, 0]])
        assert np.all(te.toarray() == [[0, 1, 0]])


def test_DataSampler():
    """Test the DataSampler class
    """
    values = [1., 1., 1., 1.]
    rows = [0, 0, 1, 1]
    cols = [0, 1, 1, 2]
    df_tr = pd.DataFrame(list(zip(rows, cols, values)), columns=['uid', 'iid', 'rating'])
    df_te_tr = pd.DataFrame([(0, 0, 1.)], columns=['uid', 'iid', 'rating'])
    df_te_te = pd.DataFrame([(0, 1, 1.)], columns=['uid', 'iid', 'rating'])
    uids = {0:0, 1:1}
    iids = {0:0, 1:1, 2:2}
    data = Dataset(df_tr, None, (df_te_tr, df_te_te), uids, iids)
    sampler = DataSampler(data, mode="train", batch_size=1, shuffle=False)

    assert len(sampler) == 2, "the number of batches should be 2"
    for i, (t, none) in enumerate(sampler):
        assert none is None, "the test part of the training should be None"
        assert isinstance(t, torch.FloatTensor), "t should be of type torch.Tensor"
        if i == 0:
            assert np.all(t.numpy() == np.array([1, 1, 0])), "the tensor t should be [1, 1, 0]"
        else:
            assert np.all(t.numpy() == np.array([0, 1, 1])), "the tensor t should be [0, 1, 1]"

    sampler.test()
    assert len(sampler) == 1, "the number of batches should be 1"

    for tr, te in sampler:
        assert np.all(tr.numpy() == np.array([1, 0, 0])), "the tensor tr should be [1, 0, 0]"
        assert np.all(te.numpy() == np.array([0, 1, 0])), "the tensor te should be [1, 1, 0]"

def test_ConditionedDataSampler():
    """Test the ConditionedDataSampler class
    """
    values = [1., 1., 1., 1.]
    rows = [0, 0, 1, 1]
    cols = [0, 1, 1, 2]
    df_tr = pd.DataFrame(list(zip(rows, cols, values)), columns=['uid', 'iid', 'rating'])
    df_te_tr = pd.DataFrame([(0, 0, 1.)], columns=['uid', 'iid', 'rating'])
    df_te_te = pd.DataFrame([(0, 1, 1.)], columns=['uid', 'iid', 'rating'])
    uids = {0:0, 1:1}
    iids = {0:0, 1:1, 2:2}
    data = Dataset(df_tr, None, (df_te_tr, df_te_te), uids, iids)
    iid2cids = {0:[1], 1:[0, 1], 2:[0]}
    sampler = ConditionedDataSampler(iid2cids, 2, data, mode="train", batch_size=2, shuffle=False)

    assert len(sampler) == 3, "the number of batches should be 2"
    for i, (tr, te) in enumerate(sampler):
        assert isinstance(tr, torch.FloatTensor), "tr should be of type torch.Tensor"
        assert isinstance(te, torch.FloatTensor), "te should be of type torch.Tensor"
        if i == 0:
            assert np.all(tr.numpy() == np.array([[1, 1, 0, 0, 0], [0, 1, 1, 0, 0]])),\
                "the tensor tr should be [[1, 1, 0, 0, 0], [0, 1, 1, 0, 0]]"
            assert np.all(te.numpy() == np.array([[1, 1, 0], [0, 1, 1]])),\
                "the tensor te should be [[1, 1, 0], [0, 1, 1]]"
        elif i == 1:
            assert np.all(tr.numpy() == np.array([[1, 1, 0, 1, 0], [1, 1, 0, 0, 1]])),\
                "the tensor tr should be [[1, 1, 0, 1, 0], [1, 1, 0, 0, 1]]"
            assert np.all(te.numpy() == np.array([[0, 1, 0], [1, 1, 0]])),\
                "the tensor te should be [[1, 0, 1], [1, 1, 0]]"
        else:
            assert np.all(tr.numpy() == np.array([[0, 1, 1, 1, 0], [0, 1, 1, 0, 1]])),\
                "the tensor tr should be [[0, 1, 1, 1, 0], [0, 1, 1, 0, 1]]"
            assert np.all(te.numpy() == np.array([[0, 1, 1], [0, 1, 0]])),\
                "the tensor te should be [[0, 1, 1], [0, 1, 0]]"

    np.random.seed(1)
    sampler.test(1)
    print(sampler.batch_size)
    print(sampler.mode)
    print(sampler.sparse_data_tr)
    print(sampler.sparse_data_te)
    assert len(sampler) == 2, "the number of batches should be 2"
    for i, (tr, te) in enumerate(sampler):
        assert isinstance(tr, torch.FloatTensor), "tr should be of type torch.Tensor"
        assert isinstance(te, torch.FloatTensor), "te should be of type torch.Tensor"
        if i == 0:
            assert np.all(tr.numpy() == np.array([1, 0, 0, 0, 0])),\
                "the tensor tr should be [1, 0, 0, 0, 0]"
            assert np.all(te.numpy() == np.array([0, 1, 0])),\
                "the tensor te should be [0, 1, 0]"
        else:
            assert np.all(tr.numpy() == np.array([1, 0, 0, 0, 1])),\
                "the tensor tr should be [1, 0, 0, 0, 1]"
            assert np.all(te.numpy() == np.array([0, 1, 0])),\
                "the tensor te should be [0, 1, 0]"

def test_EmptyConditionedDataSampler():
    """Test the EmptyConditionedDataSampler class
    """
    values = [1., 1., 1., 1.]
    rows = [0, 0, 1, 1]
    cols = [0, 1, 1, 2]
    df_tr = pd.DataFrame(list(zip(rows, cols, values)), columns=['uid', 'iid', 'rating'])
    df_te_tr = pd.DataFrame([(0, 0, 1.)], columns=['uid', 'iid', 'rating'])
    df_te_te = pd.DataFrame([(0, 1, 1.)], columns=['uid', 'iid', 'rating'])
    uids = {0:0, 1:1}
    iids = {0:0, 1:1, 2:2}
    data = Dataset(df_tr, None, (df_te_tr, df_te_te), uids, iids)
    sampler = EmptyConditionedDataSampler(2, data, mode="train", batch_size=2, shuffle=False)

    assert len(sampler) == 1, "the number of batches should be 1"
    for tr, te in sampler:
        assert isinstance(tr, torch.FloatTensor), "tr should be of type torch.Tensor"
        assert isinstance(te, torch.FloatTensor), "te should be of type torch.Tensor"
        assert np.all(tr.numpy() == np.array([[1, 1, 0, 0, 0], [0, 1, 1, 0, 0]])),\
            "the tensor tr should be [[1, 1, 0, 0, 0], [0, 1, 1, 0, 0]]"
        assert np.all(te.numpy() == np.array([[1, 1, 0], [0, 1, 1]])),\
            "the tensor te should be [[1, 1, 0], [0, 1, 1]]"

    np.random.seed(1)
    sampler.test(1)
    assert len(sampler) == 1, "the number of batches should be 1"
    for tr, te in sampler:
        assert isinstance(tr, torch.FloatTensor), "tr should be of type torch.Tensor"
        assert isinstance(te, torch.FloatTensor), "te should be of type torch.Tensor"
        assert np.all(tr.numpy() == np.array([1, 0, 0, 0, 0])),\
            "the tensor tr should be [1, 0, 0, 0, 0]"
        assert np.all(te.numpy() == np.array([0, 1, 0])),\
            "the tensor te should be [0, 1, 0]"

def test_NCR_Sampler():
    """Test the NCR_Sampler class
    """
    set_seed(2021)
    tmp = tempfile.NamedTemporaryFile()
    with open(tmp.name, "w") as f:
        f.write("userID itemID rating timestamp\n")
        f.write("1 8 4 4\n1 6 1 3\n1 9 2 5\n1 5 4 2\n1 10 5 6\n1 7 3 1\n")
        f.write("2 6 5 1\n2 1 3 3\n2 3 2 4\n2 4 5 5\n2 5 4 6\n2 8 4 2\n2 7 3 7\n")

    with tempfile.TemporaryDirectory():
        tmp_d = tempfile.NamedTemporaryFile()
        cfg_d = {
            "processing": {
                "data_path": tmp.name,
                "separator": " ",
                "header": 0,
                "rating_order": 1,
                "rating_threshold": 4,
                "max_history_length": 3,
                "premise_threshold": 0
            },
            "splitting": {
                "leave_n": 1,
                "keep_n": 2
            }
        }
        json.dump(cfg_d, open(tmp_d.name, "w"))

        dp = NCRDataProcessing(tmp_d.name)
        data = dp.process_and_split()

        # test train mode

        # test sampler with shuffle

        sampler = NCR_Sampler(data, mode="train", batch_size=1, n_neg_samples_t=1, n_neg_samples_vt=2, shuffle=True)

        assert len(sampler) == 3, "the number of batches should be 3"
        for i, (user_ids, item_ids, histories, feedbacks, negative_item_ids) in enumerate(sampler):
            assert isinstance(user_ids, torch.Tensor), "user_ids should be of type torch.Tensor"
            assert isinstance(item_ids, torch.Tensor), "item_ids should be of type torch.Tensor"
            assert isinstance(histories, torch.Tensor), "histories should be of type torch.Tensor"
            assert isinstance(feedbacks, torch.Tensor), "feedbacks should be of type torch.Tensor"
            assert isinstance(negative_item_ids, torch.Tensor), "negative_item_ids should be of type torch.Tensor"

            if i == 0:
                assert user_ids.numpy()[0] == 1
                assert item_ids.numpy()[0] == 0
                assert histories.numpy()[0] == 1
                assert feedbacks.numpy()[0] == 1
                assert negative_item_ids.numpy()[0] == 4
            if i == 1:
                assert user_ids.numpy()[0] == 0
                assert item_ids.numpy()[0] == 3
                assert histories.numpy()[0] == 5
                assert feedbacks.numpy()[0] == 0
                assert negative_item_ids.numpy()[0] == 8
            if i == 2:
                assert user_ids.numpy()[0] == 0
                assert item_ids.numpy()[0] == 0
                assert (histories.numpy()[0] == np.array([5, 3, 1])).all()
                assert (feedbacks.numpy()[0] == np.array([0, 1, 0])).all()
                assert negative_item_ids.numpy()[0] == 8

        # test sampler without shuffle, change also batch size in order to test it

        sampler = NCR_Sampler(data, mode="train", batch_size=2, n_neg_samples_t=1, n_neg_samples_vt=2,
                              shuffle=False)

        assert len(sampler) == 2, "the number of batches should be 2"
        for i, (user_ids, item_ids, histories, feedbacks, negative_item_ids) in enumerate(sampler):
            assert isinstance(user_ids, torch.Tensor), "user_ids should be of type torch.Tensor"
            assert isinstance(item_ids, torch.Tensor), "item_ids should be of type torch.Tensor"
            assert isinstance(histories, torch.Tensor), "histories should be of type torch.Tensor"
            assert isinstance(feedbacks, torch.Tensor), "feedbacks should be of type torch.Tensor"
            assert isinstance(negative_item_ids,
                              torch.Tensor), "negative_item_ids should be of type torch.Tensor"

            if i == 0:
                assert (user_ids.numpy() == np.array([0, 1])).all()
                assert (item_ids.numpy() == np.array([3, 0])).all()
                assert (histories.numpy() == np.array([[5], [1]])).all()
                assert (feedbacks.numpy() == np.array([[0], [1]])).all()
                assert (negative_item_ids.numpy() == np.array([[7], [2]])).all()
            if i == 1:
                assert user_ids.numpy()[0] == 0
                assert item_ids.numpy()[0] == 0
                assert (histories.numpy()[0] == np.array([5, 3, 1])).all()
                assert (feedbacks.numpy()[0] == np.array([0, 1, 0])).all()
                assert negative_item_ids.numpy()[0] == 8

        # test valid mode

        sampler.valid(batch_size=1)

        assert len(sampler) == 1, "the number of batches should be 1"

        for user_ids, item_ids, histories, feedbacks, negative_item_ids in sampler:
            assert isinstance(user_ids, torch.Tensor), "user_ids should be of type torch.Tensor"
            assert isinstance(item_ids, torch.Tensor), "item_ids should be of type torch.Tensor"
            assert isinstance(histories, torch.Tensor), "histories should be of type torch.Tensor"
            assert isinstance(feedbacks, torch.Tensor), "feedbacks should be of type torch.Tensor"
            assert isinstance(negative_item_ids,
                              torch.Tensor), "negative_item_ids should be of type torch.Tensor"

            assert user_ids.numpy()[0] == 1
            assert item_ids.numpy()[0] == 8
            assert (histories.numpy() == np.array([0, 6, 7])).all()
            assert (feedbacks.numpy() == np.array([1, 0, 0])).all()
            assert (negative_item_ids.numpy() == np.array([2, 4])).all()

        # test test mode

        sampler.test()
        assert len(sampler) == 2, "the number of batches should be 2"

        for i, (user_ids, item_ids, histories, feedbacks, negative_item_ids) in enumerate(sampler):
            assert isinstance(user_ids, torch.Tensor), "user_ids should be of type torch.Tensor"
            assert isinstance(item_ids, torch.Tensor), "item_ids should be of type torch.Tensor"
            assert isinstance(histories, torch.Tensor), "histories should be of type torch.Tensor"
            assert isinstance(feedbacks, torch.Tensor), "feedbacks should be of type torch.Tensor"
            assert isinstance(negative_item_ids,
                              torch.Tensor), "negative_item_ids should be of type torch.Tensor"

            if i == 0:
                assert user_ids.numpy()[0] == 0
                assert item_ids.numpy()[0] == 4
                assert (histories.numpy() == np.array([1, 0, 2])).all()
                assert (feedbacks.numpy() == np.array([0, 1, 0])).all()
                assert (negative_item_ids.numpy() == np.array([7, 6])).all()
            if i == 1:
                assert user_ids.numpy()[0] == 1
                assert item_ids.numpy()[0] == 3
                assert (histories.numpy() == np.array([6, 7, 8])).all()
                assert (feedbacks.numpy() == np.array([0, 0, 1])).all()
                assert (negative_item_ids.numpy() == np.array([4, 2])).all()


def test_CFGAN_Sampler():
    """Test the CFGAN_Sampler class
    """
    values = [1., 1., 1., 1.]
    rows = [0, 0, 1, 1]
    cols = [0, 1, 1, 2]
    df_tr = pd.DataFrame(list(zip(rows, cols, values)), columns=['uid', 'iid', 'rating'])
    df_te_tr = pd.DataFrame([(0, 0, 1.)], columns=['uid', 'iid', 'rating'])
    df_te_te = pd.DataFrame([(0, 1, 1.)], columns=['uid', 'iid', 'rating'])
    uids = {0:0, 1:1}
    iids = {0:0, 1:1, 2:2}
    data = Dataset(df_tr, None, (df_te_tr, df_te_te), uids, iids)
    sampler = CFGAN_Sampler(data, mode="train", batch_size=1)

    assert len(sampler) == 2, "the number of batches should be 2"
    assert hasattr(sampler, "idxlist"), "the sampler should have the attribute idxlist"
    #assert sampler.idxlist == [0, 1], "the idxlist should be only [0, 1]"

    t = None
    for x in sampler:
        t = x
        break

    assert isinstance(t, torch.FloatTensor), "t should be of type torch.Tensor"
    assert np.all(t.numpy() == np.array([1, 1, 0])) or np.all(t.numpy() == np.array([0, 1, 1])),\
        "the next batch should be [1, 1, 0] or [0, 1, 1]"

    t = next(iter(sampler))
    assert np.all(t.numpy() == np.array([1, 1, 0])) or np.all(t.numpy() == np.array([0, 1, 1])),\
        "the next batch should be [1, 1, 0] or [0, 1, 1]"

def test_SVAE_Sampler():
    """Test the SVAE_Sampler class
    """
    tr = {0:[0, 1, 2, 3, 4, 5, 6], 1:[6, 5, 4, 3, 2, 1, 0], 2:[2, 1, 6, 0, 3]}

    values = [1.] * 19
    rows = [0] * 7 + [1] * 7 + [2] * 5
    cols = list(range(7)) + list(range(6, -1, -1)) + [2, 1, 6, 0, 3]
    tt = list(range(7)) + list(range(7)) + list(range(5))
    df_tr = pd.DataFrame(list(zip(rows, cols, values, tt)),
                         columns=['uid', 'iid', 'rating', 'time'])

    values = [1.] * 10
    rows = [0] * 4 + [1] * 4 + [2] * 2
    cols = [0, 1, 2, 3, 6, 5, 4, 3, 1, 6]
    tt = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1]
    df_te_tr = pd.DataFrame(list(zip(rows, cols, values, tt)),
                            columns=['uid', 'iid', 'rating', 'time'])

    values = [1.] * 8
    rows = [0, 0, 0, 1, 1, 1, 2, 2]
    cols = [4, 5, 6, 2, 1, 0, 0, 3]
    tt = [0, 1, 2, 0, 1, 2, 0, 1]
    df_te_te = pd.DataFrame(list(zip(rows, cols, values, tt)),
                            columns=['uid', 'iid', 'rating', 'time'])

    uids = {0:0, 1:1, 2:2}
    iids = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6}
    data = Dataset(df_tr, None, (df_te_tr, df_te_te), uids, iids)

    sampler = SVAE_Sampler(data, mode="train",
                           pred_type="next_k",
                           k=2,
                           shuffle=False)

    assert len(sampler) == 3
    assert hasattr(sampler, "num_items")
    assert hasattr(sampler, "k")
    assert hasattr(sampler, "shuffle")
    assert hasattr(sampler, "pred_type")
    assert hasattr(sampler, "dict_data_tr")
    assert hasattr(sampler, "dict_data_te")
    assert sampler.k == 2
    assert sampler.num_items == 7
    assert sampler.pred_type == "next_k"
    assert not sampler.shuffle

    i = 0
    res = [np.array([[[0, 1, 1, 0, 0, 0, 0],
                      [0, 0, 1, 1, 0, 0, 0],
                      [0, 0, 0, 1, 1, 0, 0],
                      [0, 0, 0, 0, 1, 1, 0],
                      [0, 0, 0, 0, 0, 1, 1],
                      [0, 0, 0, 0, 0, 0, 1]]]),
           np.array([[[0, 0, 0, 0, 1, 1, 0],
                      [0, 0, 0, 1, 1, 0, 0],
                      [0, 0, 1, 1, 0, 0, 0], 
                      [0, 1, 1, 0, 0, 0, 0],
                      [1, 1, 0, 0, 0, 0, 0],
                      [1, 0, 0, 0, 0, 0, 0]]]),
           np.array([[[0, 1, 0, 0, 0, 0, 1],
                      [1, 0, 0, 0, 0, 0, 1],
                      [1, 0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0, 0]]])]
    for x, y in sampler:
        assert isinstance(x, torch.LongTensor), "x should be of type torch.LongTensor"
        assert isinstance(y, torch.FloatTensor), "y should be of type torch.FloatTensor"
        assert np.all(x.numpy() == tr[i][:-1])
        if i == 2:
            assert y.shape == (1, 4, 7)
        else:
            assert y.shape == (1, 6, 7)
        assert np.all(y.numpy() == res[i])
        i += 1

    sampler = SVAE_Sampler(data, mode="train",
                           pred_type="next",
                           k=2,
                           shuffle=False)

    i = 0
    res = [np.array([[[0, 1, 0, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 0, 1]]]),
           np.array([[[0, 0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 1, 0, 0, 0], 
                      [0, 0, 1, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0, 0],
                      [1, 0, 0, 0, 0, 0, 0]]]),
           np.array([[[0, 1, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 1],
                      [1, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0, 0]]])]
    for x, y in sampler:
        assert isinstance(x, torch.LongTensor), "x should be of type torch.LongTensor"
        assert isinstance(y, torch.FloatTensor), "y should be of type torch.FloatTensor"
        assert np.all(x.numpy() == tr[i][:-1])
        if i == 2:
            assert y.shape == (1, 4, 7)
        else:
            assert y.shape == (1, 6, 7)
        assert np.all(y.numpy() == res[i])
        i += 1

    sampler = SVAE_Sampler(data, mode="train",
                           pred_type="postfix",
                           k=2,
                           shuffle=False)

    i = 0
    res = [np.array([[[0, 1, 1, 1, 1, 1, 1],
                      [0, 0, 1, 1, 1, 1, 1],
                      [0, 0, 0, 1, 1, 1, 1],
                      [0, 0, 0, 0, 1, 1, 1],
                      [0, 0, 0, 0, 0, 1, 1],
                      [0, 0, 0, 0, 0, 0, 1]]]),
           np.array([[[1, 1, 1, 1, 1, 1, 0],
                      [1, 1, 1, 1, 1, 0, 0],
                      [1, 1, 1, 1, 0, 0, 0],
                      [1, 1, 1, 0, 0, 0, 0],
                      [1, 1, 0, 0, 0, 0, 0],
                      [1, 0, 0, 0, 0, 0, 0]]]),
           np.array([[[1, 1, 0, 1, 0, 0, 1],
                      [1, 0, 0, 1, 0, 0, 1],
                      [1, 0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0, 0]]])]
    for x, y in sampler:
        assert isinstance(x, torch.LongTensor), "x should be of type torch.LongTensor"
        assert isinstance(y, torch.FloatTensor), "y should be of type torch.FloatTensor"
        assert np.all(x.numpy() == tr[i][:-1])
        if i == 2:
            assert y.shape == (1, 4, 7)
        else:
            assert y.shape == (1, 6, 7)
        assert np.all(y.numpy() == res[i])
        i += 1

    sampler = SVAE_Sampler(data, mode="test",
                           pred_type="next_k",
                           k=2,
                           shuffle=False)

    i = 0
    res = [np.array([[[0, 0, 0, 0, 1, 1, 1]]]),
           np.array([[[1, 1, 1, 0, 0, 0, 0]]]),
           np.array([[[1, 0, 0, 1, 0, 0, 0]]])]
    for x, y in sampler:
        assert isinstance(x, torch.LongTensor), "x should be of type torch.LongTensor"
        assert isinstance(y, torch.FloatTensor), "y should be of type torch.FloatTensor"
        vtr = {0:[0, 1, 2, 3], 1:[6, 5, 4, 3], 2:[1, 6]}
        assert np.all(x.numpy() == vtr[i][:-1])
        assert y.shape == (1, 1, 7)
        assert np.all(y.numpy() == res[i])
        i += 1

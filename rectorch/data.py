r"""The ``data`` module manages the reading, writing and loading of the data sets.

The supported data set format is standard `csv
<https://it.wikipedia.org/wiki/Comma-separated_values>`_.
For more information about the expected data set format please visit :ref:`csv-format`.
The data processing and loading configurations are managed through the configuration files
as described in :ref:`config-format`. Please, note that the Neural Collaborative Reasoning (NCR)
approach requires a different pre-processing procedure and configuration files compared to the
other methods implemented in rectorch. Refer to :ref:`config-format` to find detailed information.
The vertical data splitting phase is highly inspired by `VAE-CF source code
<https://github.com/dawenl/vae_cf>`_, which has been lately used on several other research works.

Examples
--------
This module is mainly meant to be used in the following way:

>>> from rectorch.data import DataProcessing
>>> proc = DataProcessing("/path/to/the/config/file")
>>> df = proc.process()
>>> dataset = proc.split(df)

The same computation can be simplified as follows:

>>> from rectorch.data import DataProcessing
>>> dataset = DataProcessing("/path/to/the/config/file").process_and_split()

Notes
----------
For the Neural Collaborative Reasoning (NCR) model, :class:`NCRDataProcessing` has to be used instead
of :class:`DataProcessing` for the pre-processing step. Example:

>>> from rectorch.data import NCRDataProcessing
>>> dataset = NCRDataProcessing("/path/to/the/config/file").process_and_split()

See Also
--------
Modules:
:mod:`configuration <rectorch.configuration>`
"""
import os
import copy
import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy.sparse import csr_matrix, vstack
import torch
from rectorch import env
from rectorch.configuration import DataConfig

# AUTHORSHIP
__version__ = "0.9.0dev"
__author__ = "Mirko Polato"
__copyright__ = "Copyright 2020, rectorch"
__license__ = "MIT"
__maintainer__ = "Mirko Polato"
__email__ = "mak1788@gmail.com"
__status__ = "Development"
#

__all__ = ['DataProcessing', 'NCRDataProcessing', 'Dataset', 'NCRDataset']


class Dataset():
    r"""RecSys dataset.

    Dataset containing training, [validation], and test set.

    Parameters
    ----------
    uids : :obj:`list` of :obj:`int`
        List of user ids.
    iids : :obj:`list` of :obj:`int`
        List of item ids.
    train_set : :class:`pandas.DataFrame`
        The training set data frame.
    valid_set : :obj:`None` or :class:`pandas.DataFrame` or sequence of :class:`pandas.DataFrame`
        The validation set data frame. When the dataset is vertically splitted the validation
        set is a pair of data frames that correspond to the training part and the test
        part of the set. When it is set to :obj:`None` it means that no validation
        set has been created.
    test_set : :class:`pandas.DataFrame` or sequence of :class:`pandas.DataFrame`
        The test set data frame. When the dataset is vertically splitted the test
        set is a pair of data frames that correspond to the training part and the test
        part of the set.
    numerize : :obj:`bool` [optional]
        Whether the user/item ids must be re-mapped, by default :obj:`True`.

    Attributes
    ----------
    n_users : :obj:`int`
        The number of users.
    n_items : :obj:`int`
        The number of items.
    unique_uid : :obj:`list` of :obj:`int`
        List of user ids. It is the list version of :attr:`u2id`.
    unique_iid : :obj:`list` of :obj:`int`
        List of item ids. It is the list version of :attr:`i2id`.
    u2id : :obj:`dict` { :obj:`str` \: :obj:`int` }
        Dictionary which maps the raw user id, i.e., as in the raw `csv` file, to an internal id
        which is an integer between 0 and the total number of users minus one.
    i2id : :obj:`dict` { :obj:`str` \: :obj:`int` }
        Dictionary which maps the raw item id, i.e., as in the raw `csv` file, to an internal id
        which is an integer between 0 and the total number of items minus one.
    train_set : :class:`pandas.DataFrame`
        See ``train_set`` parameter.
    valid_set : :obj:`None` or :class:`pandas.DataFrame` or tuple of :class:`pandas.DataFrame`
        See ``valid_set`` parameter.
    test_set : :class:`pandas.DataFrame` or tuple of :class:`pandas.DataFrame`
        See ``test_set`` parameter.
    """

    def __init__(self, train_set, valid_set, test_set, uids, iids, numerize=True):
        assert isinstance(train_set, DataFrame), "train_set must be a DataFrame"
        if valid_set is not None:
            assert isinstance(valid_set, (DataFrame, tuple, list, np.ndarray)), \
                "valid_set must be a DataFrame or a tuple/list/array of DataFrames"
        assert isinstance(test_set, (DataFrame, tuple, list, np.ndarray)), \
            "test_set must be a DataFrame or a tuple of DataFrames"

        if not isinstance(valid_set, DataFrame) and valid_set is not None:
            assert len(valid_set) == 2, "valid_set must be a sequence of DataFrames of length 2"
            if valid_set[0] is not None:
                assert isinstance(valid_set[0], DataFrame) and isinstance(valid_set[1], DataFrame), \
                    "valid_set must be a sequence of DataFrames"
            else:
                valid_set = None

        self.unique_uid = uids
        self.unique_iid = iids
        self.n_users = len(uids)
        self.n_items = len(iids)
        self.u2id, self.i2id = self._mapping()

        if numerize:
            self.train_set = self._numerize(train_set)
            if valid_set is not None:
                if isinstance(valid_set, DataFrame):
                    self.valid_set = self._numerize(valid_set)
                else:
                    self.valid_set = [self._numerize(v) for v in valid_set]
            else:
                self.valid_set = None
            if isinstance(test_set, DataFrame):
                self.test_set = self._numerize(test_set)
            else:
                self.test_set = [self._numerize(t) for t in test_set]
        else:
            self.train_set = train_set
            self.valid_set = valid_set
            self.test_set = test_set

        self.n_ratings = len(self.train_set)
        if self.valid_set is not None:
            if isinstance(self.valid_set, DataFrame):
                self.n_ratings += len(self.valid_set)
            else:
                self.n_ratings += len(self.valid_set[0]) + len(self.valid_set[1])

        if isinstance(self.test_set, DataFrame):
            self.n_ratings += len(self.test_set)
        else:
            self.n_ratings += len(self.test_set[0]) + len(self.test_set[1])

    def _mapping(self):
        u2id = dict((uid, i) for (i, uid) in enumerate(self.unique_uid))
        i2id = dict((iid, i) for (i, iid) in enumerate(self.unique_iid))
        return u2id, i2id

    def _numerize(self, data):
        uhead, ihead = data.columns.values[:2]
        uid = data[uhead].apply(lambda x: self.u2id[x])
        iid = data[ihead].apply(lambda x: self.i2id[x])
        dic_data = {'uid': uid, 'iid': iid, 'rating': data[data.columns.values[2]]}
        for c in data.columns.values[3:]:
            dic_data[c] = data[c]
        cols = ['uid', 'iid', 'rating'] + list(data.columns[3:])
        return pd.DataFrame(data=dic_data, columns=cols)

    def save(self, pro_dir):
        r"""Save the dataset.

        The dataset is saved as a series of files that changes on the basis of the
        nature of the dataset. Specifically, the output consists of a series of
        files saved in ``pro_dir``:

        * ``train.csv`` : (`csv` file) the training ratings corresponding to all ratings of the\
          training users;
        * ``validation_tr.csv`` (*) : (`csv` file) the ratings corresponding to the validation set.
        * ``validation_tr.csv`` (**) : (`csv` file) the training ratings corresponding to the\
          validation users.
        * ``validation_te.csv`` (**) : (`csv` file) the test ratings corresponding to the\
          validation users;
        * ``test_tr.csv`` (*) : (`csv` file) the ratings corresponding to the test set;
        * ``test_tr.csv`` (**): (`csv` file) the training ratings corresponding to the test users;
        * ``test_te.csv`` (**): (`csv` file) the test ratings corresponding to the test users;
        * ``unique_uid.txt`` : (`txt` file) with the user id mapping. Line numbers represent the\
          internal id, while the string on the corresponding line is the raw id;
        * ``unique_iid.txt`` : (`txt` file) with the item id mapping. Line numbers represent the\
          internal id, while the string on the corresponding line is the raw id;

        Where (*) means that the file is created only in the case of horizontal splitting, and
        (**) means that the file is created only in the case of vertical splitting.


        Parameters
        ----------
        pro_dir : :obj:`str`
            Path to the folder where files will be saved.
        """
        env.logger.info("Saving unique_iid.txt.")
        if not os.path.exists(pro_dir):
            os.makedirs(pro_dir)

        with open(os.path.join(pro_dir, 'unique_iid.txt'), 'w') as f:
            for iid in self.unique_iid:
                f.write('%s\n' % iid)

        env.logger.info("Saving unique_uid.txt.")
        with open(os.path.join(pro_dir, 'unique_uid.txt'), 'w') as f:
            for uid in self.unique_uid:
                f.write('%s\n' % uid)

        env.logger.info("Saving all the files.")
        self.train_set.to_csv(os.path.join(pro_dir, 'train.csv'), index=False)
        if self.valid_set:
            if isinstance(self.valid_set, DataFrame):
                self.valid_set.to_csv(os.path.join(pro_dir, 'validation.csv'), index=False)
            else:
                self.valid_set[0].to_csv(os.path.join(pro_dir, 'validation_tr.csv'), index=False)
                self.valid_set[1].to_csv(os.path.join(pro_dir, 'validation_te.csv'), index=False)

        if isinstance(self.test_set, DataFrame):
            self.test_set.to_csv(os.path.join(pro_dir, 'test.csv'), index=False)
        else:
            self.test_set[0].to_csv(os.path.join(pro_dir, 'test_tr.csv'), index=False)
            self.test_set[1].to_csv(os.path.join(pro_dir, 'test_te.csv'), index=False)

        env.logger.info("Dataset saved successfully!")

    @classmethod
    def load(cls, pro_dir):
        r"""Load the dataset.

        Load the dataset according to the files in the ``pro_dir`` folder.

        Parameters
        ----------
        pro_dir : :obj:`str`
            Path to the folder where the dataset files are stored.
        """
        unique_uid = []
        with open(os.path.join(pro_dir, 'unique_uid.txt'), 'r') as f:
            for line in f:
                unique_uid.append(line.strip())

        unique_iid = []
        with open(os.path.join(pro_dir, 'unique_iid.txt'), 'r') as f:
            for line in f:
                unique_iid.append(line.strip())

        train_data = pd.read_csv(os.path.join(pro_dir, 'train.csv'))

        val_path = os.path.join(pro_dir, 'validation.csv')
        if os.path.isfile(val_path):
            val_data = pd.read_csv(val_path)
        else:
            vtr_path = os.path.join(pro_dir, 'validation_tr.csv')
            if os.path.isfile(vtr_path):
                vte_path = os.path.join(pro_dir, 'validation_te.csv')
                val_data = [pd.read_csv(vtr_path), pd.read_csv(vte_path)]
            else:
                val_data = None

        test_path = os.path.join(pro_dir, 'test.csv')
        if os.path.isfile(test_path):
            test_data = pd.read_csv(test_path)
        else:
            ttr_path = os.path.join(pro_dir, 'test_tr.csv')
            tte_path = os.path.join(pro_dir, 'test_te.csv')
            test_data = [pd.read_csv(ttr_path), pd.read_csv(tte_path)]

        return Dataset(train_data, val_data, test_data, unique_uid, unique_iid, numerize=False)

    def to_dict(self, binarize=True, cold_users=True):
        r"""Convert the dataset to dictionaries

        The dataset is converted into a series of dictionaries. Each dictionary
        has users as keys and list of items as values.

        Parameters
        ----------
        binarize : :obj:`bool`
            Whether the ratings have to be binarized.
        cold_users : :obj:`bool` [optional]
            Whether cold users are included in the validation/test, by default :obj:`True`.
            Note: it is used only when the dataset is vertically splitted.

        Returns
        -------
        data_tr : :obj:`dict` { :obj:`int` \: :obj:`int` }
            Dictionary containing the training ratings.
        data_val : [sequence of] :obj:`dict` { :obj:`int` \: :obj:`int` } or :obj:`None`
            If the dataset is horizontally splitted it is a dictionary containing
            the validation ratings. Otherwise, it is a pair of dictionaries containing
            the training and test part of the validation ratings. In case of no validation
            set it is a :obj:`None` object.
        data_te : [sequence of] :obj:`dict` { :obj:`int` \: :obj:`int` }
            If the dataset is horizontally splitted it is a dictionary containing
            the test ratings. Otherwise, it is a pair of dictionaries containing
            the training and test part of the test ratings.
        """
        data_tr = self._to_dict(self.train_set, binarize)
        data_val = None
        if isinstance(self.test_set, DataFrame):
            if self.valid_set is not None:
                data_val = self._to_dict(self.valid_set, binarize)
            data_te = self._to_dict(self.test_set, binarize)
        else:
            if cold_users:
                if self.valid_set is not None:
                    data_val = tuple([self._to_dict(v, binarize) for v in self.valid_set])
                data_te = tuple([self._to_dict(t, binarize) for t in self.test_set])
            else:
                if self.valid_set is not None:
                    data_tr.update(self._to_dict(self.valid_set[0], binarize))
                    data_val = self._to_dict(self.valid_set[1], binarize)
                data_tr.update(self._to_dict(self.test_set[0], binarize))
                data_te = self._to_dict(self.test_set[1], binarize)
        return data_tr, data_val, data_te

    def _to_dict(self, data, binarize):
        grouped = data.groupby(by="uid")
        if binarize:
            return {idx: list(group["iid"]) for idx, group in grouped}
        else:
            return {idx: zip(list(gr["iid"]), list(gr["rating"])) for idx, gr in grouped}

    def to_array(self, binarize=True, cold_users=True):
        r"""Return the dataset as a numpy array.

        The dataset is returned as a tuple according to the way it is splitted.

        Parameters
        ----------
        binarize : :obj:`bool` [optional]
            Whether the ratings have to be binarize or not, by default :obj:`True`.
        cold_users : :obj:`bool` [optional]
            Whether cold users are included in the validation/test, by default :obj:`True`.
            Note: it is used only when the dataset is vertically splitted.

        Returns
        -------
        :obj:`tuple` of :class:`numpy.ndarray`
            In case of horizonal splitting it returns (training set, validation set,
            test set). In case of vertical splitting it returns (training set,
            (training part of the validation set, test part of the validation set),
            (training part of the test set, test part of the test set))..
        """
        data_tr = self._df_to_array(self.train_set, binarize)
        data_val = None
        if isinstance(self.test_set, DataFrame):
            data_val = None
            if self.valid_set is not None:
                data_val = self._df_to_array(self.valid_set, binarize)
            data_te = self._df_to_array(self.test_set, binarize)
        else:
            if cold_users:
                if self.valid_set is not None:
                    data_val = self._seq_to_array(self.valid_set, binarize)
                data_te = self._seq_to_array(self.test_set, binarize)
            else:
                if self.valid_set is not None:
                    data_val = self._df_to_array(self.valid_set[1], binarize)
                    dval_tr = self._df_to_array(self.valid_set[0], binarize)
                    data_tr = np.concatenate((data_tr, dval_tr))
                data_te = self._df_to_array(self.test_set[1], binarize)
                dte_tr = self._df_to_array(self.test_set[0], binarize)
                data_tr = np.concatenate((data_tr, dte_tr))
        return data_tr, data_val, data_te

    def _df_to_array(self, data, binarize):
        start_idx = data['uid'].min()
        rows, cols = data['uid'] - start_idx, data['iid']
        n_tr_users = data['uid'].max() - start_idx + 1
        array = np.zeros((n_tr_users, self.n_items))
        array[rows, cols] = 1. if binarize else data[data.columns.values[2]]
        return array

    def _seq_to_array(self, data, binarize):
        data_tr, data_te = data[0], data[1]

        start_idx = min(data_tr['uid'].min(), data_te['uid'].min())
        end_idx = max(data_tr['uid'].max(), data_te['uid'].max())

        rows_tr, cols_tr = data_tr['uid'] - start_idx, data_tr['iid']
        rows_te, cols_te = data_te['uid'] - start_idx, data_te['iid']

        array_tr = np.zeros((end_idx - start_idx + 1, self.n_items))
        array_tr[rows_tr, cols_tr] = 1. if binarize else data_tr[data_tr.columns.values[2]]

        array_te = np.zeros((end_idx - start_idx + 1, self.n_items))
        array_te[rows_te, cols_te] = 1. if binarize else data_te[data_te.columns.values[2]]

        tr_idx = array_tr.any(axis=1)
        return array_tr[tr_idx], array_te[tr_idx]

    def to_sparse(self, binarize=True, cold_users=True):
        r"""Return the dataset as a scipy sparse csr_matrix.

        The dataset is returned as a tuple according to the way it is splitted.

        Parameters
        ----------
        binarize : :obj:`bool` [optional]
            Whether the ratings have to be binarize or not, by default :obj:`True`.
        cold_users : :obj:`bool` [optional]
            Whether cold users are included in the validation/test, by default :obj:`True`.
            Note: it is used only when the dataset is vertically splitted.

        Returns
        -------
        :obj:`tuple` of :class:`scipy.sparse.csr_matrix`
            In case of horizonal splitting it returns (training set, validation set,
            test set). In case of vertical splitting it returns (training set,
            (training part of the validation set, test part of the validation set),
            (training part of the test set, test part of the test set))..
        """
        data_tr = self._df_to_sparse(self.train_set, binarize)
        data_val = None
        if isinstance(self.test_set, DataFrame):
            if self.valid_set is not None:
                data_val = self._df_to_sparse(self.valid_set, binarize)
            data_te = self._df_to_sparse(self.test_set, binarize)
        else:
            if cold_users:
                if self.valid_set is not None:
                    data_val = self._seq_to_sparse(self.valid_set, binarize)
                data_te = self._seq_to_sparse(self.test_set, binarize)
            else:
                if self.valid_set is not None:
                    data_val = self._df_to_sparse(self.valid_set[1], binarize)
                    dval_tr = self._df_to_sparse(self.valid_set[0], binarize)
                    data_tr = vstack([data_tr, dval_tr])
                data_te = self._df_to_sparse(self.test_set[1], binarize)
                dte_tr = self._df_to_sparse(self.test_set[0], binarize)
                data_tr = vstack([data_tr, dte_tr])
        return data_tr, data_val, data_te

    def _df_to_sparse(self, data, binarize):
        start_idx = data['uid'].min()
        rows, cols = data['uid'] - start_idx, data['iid']
        n_tr_users = data['uid'].max() - start_idx + 1
        values = np.ones_like(rows) if binarize else data[data.columns.values[2]]
        return csr_matrix((values, (rows, cols)),
                          dtype='float64',
                          shape=(n_tr_users, self.n_items))

    def _seq_to_sparse(self, data, binarize):
        data_tr, data_te = data[0], data[1]

        start_idx = min(data_tr['uid'].min(), data_te['uid'].min())
        end_idx = max(data_tr['uid'].max(), data_te['uid'].max())

        rows_tr, cols_tr = data_tr['uid'] - start_idx, data_tr['iid']
        rows_te, cols_te = data_te['uid'] - start_idx, data_te['iid']

        if binarize:
            values_tr = np.ones_like(rows_tr)
            values_te = np.ones_like(rows_te)
        else:
            values_tr = data_tr[data_tr.columns.values[2]]
            values_te = data_te[data_tr.columns.values[2]]

        data_tr = csr_matrix((values_tr, (rows_tr, cols_tr)),
                             dtype='float64',
                             shape=(end_idx - start_idx + 1, self.n_items))
        data_te = csr_matrix((values_te, (rows_te, cols_te)),
                             dtype='float64',
                             shape=(end_idx - start_idx + 1, self.n_items))

        tr_idx = np.diff(data_tr.indptr) != 0
        return data_tr[tr_idx], data_te[tr_idx]

    def to_tensor(self, binarize=True, cold_users=True):
        r"""Return the dataset as a pytorch tensor.

        The dataset is returned as a tuple according to the way it is splitted.

        Parameters
        ----------
        binarize : :obj:`bool` [optional]
            Whether the ratings have to be binarize or not, by default :obj:`True`.
        cold_users : :obj:`bool` [optional]
            Whether cold users are included in the validation/test, by default :obj:`True`.
            Note: it is used only when the dataset is vertically splitted.

        Returns
        -------
        :obj:`tuple` of :class:`torch.FloatTensor`
            In case of horizonal splitting it returns (training set, validation set,
            test set). In case of vertical splitting it returns (training set,
            (training part of the validation set, test part of the validation set),
            (training part of the test set, test part of the test set)).
        """
        data_tr = self._df_to_tensor(self.train_set, binarize)
        data_val = None
        if isinstance(self.test_set, DataFrame):
            if self.valid_set is not None:
                data_val = self._df_to_tensor(self.valid_set, binarize)
            data_te = self._df_to_tensor(self.test_set, binarize)
        else:
            if cold_users:
                if self.valid_set is not None:
                    data_val = self._seq_to_tensor(self.valid_set, binarize)
                data_te = self._seq_to_tensor(self.test_set, binarize)
            else:
                if self.valid_set is not None:
                    data_val = self._df_to_tensor(self.valid_set[1], binarize)
                    dval_tr = self._df_to_tensor(self.valid_set[0], binarize)
                    data_tr = torch.cat([data_tr, dval_tr], dim=0)
                data_te = self._df_to_tensor(self.test_set[1], binarize)
                dte_tr = self._df_to_tensor(self.test_set[0], binarize)
                data_tr = torch.cat([data_tr, dte_tr], dim=0)
        return data_tr, data_val, data_te

    def _df_to_tensor(self, data, binarize=True):
        start_idx = data['uid'].min()
        idx = torch.LongTensor([list(data['uid'] - start_idx), list(data['iid'])])
        n_tr_users = data['uid'].max() - start_idx + 1
        values = np.ones(len(data)) if binarize else data[data.columns.values[2]]
        v = torch.FloatTensor(values)
        tensor = torch.sparse.FloatTensor(idx, v, torch.Size([n_tr_users, self.n_items]))
        return tensor.to_dense()

    def _seq_to_tensor(self, data, binarize=True):
        data_tr, data_te = data[0], data[1]

        start_idx = min(data_tr['uid'].min(), data_te['uid'].min())
        end_idx = max(data_tr['uid'].max(), data_te['uid'].max())

        idx_tr = torch.LongTensor([list(data_tr['uid'] - start_idx), list(data_tr['iid'])])
        idx_te = torch.LongTensor([list(data_te['uid'] - start_idx), list(data_te['iid'])])

        if binarize:
            values_tr = np.ones(len(data_tr))
            values_te = np.ones(len(data_te))
        else:
            values_tr = data_tr[data_tr.columns.values[2]]
            values_te = data_te[data_tr.columns.values[2]]

        v_tr = torch.FloatTensor(values_tr)
        v_te = torch.FloatTensor(values_te)

        tensor_tr = torch.sparse.FloatTensor(idx_tr,
                                             v_tr,
                                             torch.Size([end_idx - start_idx + 1, self.n_items]))
        tensor_te = torch.sparse.FloatTensor(idx_te,
                                             v_te,
                                             torch.Size([end_idx - start_idx + 1, self.n_items]))

        return (tensor_tr.to_dense(), tensor_te.to_dense())

    def __str__(self):
        return "Dataset(n_users=%d, n_items=%d, n_ratings=%d)" % (self.n_users,
                                                                  self.n_items,
                                                                  self.n_ratings)

    def __repr__(self):
        return str(self)


class DataProcessing():
    r"""Base class for the pre-processing of raw data sets.

    Data sets are expected of being `csv <https://it.wikipedia.org/wiki/Comma-separated_values>`_
    files where each row represents a rating. More details about the allowed format are described
    in :ref:`csv-format`. The pre-processing is performed following the parameters settings defined
    in the data configuration file (see :ref:`config-format` for more information).

    Parameters
    ----------
    data_config : :class:`rectorch.configuration.DataConfig`, :obj:`str`: or :obj:`dict`
        Represents the data pre-processing configurations.
        When ``type(data_config) == str`` is expected to be the path to the data configuration file.
        When ``type(data_config) == dict`` is expected to be the data configuration dictionary.
        In that case a :class:`configuration.DataConfig` object is contextually created.

    Raises
    ------
    :class:`TypeError`
        Raises when the type of the input parameter is incorrect.

    Attributes
    ----------
    cfg : :class:`rectorch.configuration.DataConfig`
        The :class:`rectorch.configuration.DataConfig` object containing the pre-processing
        configurations.

    Notes
    -----
    Each new data processor must extend this base class implementing all the abstract
    methods, in particular :meth:`DataProcessing.process` and :meth:`DataProcessing.split`.
    """

    def __init__(self, data_config):
        if isinstance(data_config, DataConfig):
            self.cfg = data_config
        elif isinstance(data_config, (str, dict)):
            self.cfg = DataConfig(data_config)
        else:
            raise TypeError("'data_config' must be of type 'DataConfig', 'dict', or 'str'.")

    def process(self):
        """Process the data set raw file.

        The pre-processing relies on the configurations provided in the data configurations
        :attr:`cfg`. The full pre-processing follows a specific pipeline that depends on the data
        processor used (the meaning of each configuration parameter is defined in :ref:`config-format`).

        Returns
        -------
        :class:`pandas.DataFrame`
            The pre-processed dataset.

        Raises
        ------
        :class:`NotImplementedError`
            Raised when not implemeneted in the sub-class.
        """
        raise NotImplementedError

    def split(self, data):
        r"""Split the data set.

        The splitting relies on the configurations provided in the data configurations
        :attr:`cfg`.

        Parameters
        ----------
        data : :class:`pandas.DataFrame`
            The dataset to split.

        Returns
        -------
        :class:`Dataset` or :obj:`list` of :class:`Dataset`
            The splitted dataset(s).

        Raises
        ------
        :class:`NotImplementedError`
            Raised when not implemeneted in the sub-class.
        """
        raise NotImplementedError

    def process_and_split(self):
        r"""Process and split the dataset.
        It is the equivalent of calling ``split(process())``.

        Returns
        -------
        :class:`Dataset`
            The processed and splitted dataset.
        """
        return self.split(self.process())


class StandardDataProcessing(DataProcessing):
    r"""Class that manages the standard rectorch pre-processing of raw data sets.

    Data sets are expected of being `csv <https://it.wikipedia.org/wiki/Comma-separated_values>`_
    files where each row represents a rating. More details about the allowed format are described
    in :ref:`csv-format`. The pre-processing is performed following the parameters settings defined
    in the data configuration file (see :ref:`config-format` for more information).

    Parameters
    ----------
    data_config : :class:`rectorch.configuration.DataConfig`, :obj:`str`: or :obj:`dict`
        Represents the data pre-processing configurations.
        When ``type(data_config) == str`` is expected to be the path to the data configuration file.
        When ``type(data_config) == dict`` is expected to be the data configuration dictionary.
        In that case a :class:`configuration.DataConfig` object is contextually created.

    Raises
    ------
    :class:`TypeError`
        Raises when the type of the input parameter is incorrect.

    Attributes
    ----------
    cfg : :class:`rectorch.configuration.DataConfig`
        The :class:`rectorch.configuration.DataConfig` object containing the pre-processing
        configurations.
    """

    def __init__(self, data_config):
        super(StandardDataProcessing, self).__init__(data_config)

    def process(self):
        r"""Process the data set raw file.

        The pre-processing relies on the configurations provided in the data configurations
        :attr:`cfg`. The full pre-processing follows a specific pipeline (the meaning of
        each configuration parameter is defined in :ref:`config-format`):

        1. Read the CSV file named ``data_path``;
        2. Filter the ratings on the basis of the ``threshold``;
        3. Filter the users and items according to ``u_min`` and ``i_min``, respectively;

        Returns
        -------
        :class:`pandas.DataFrame`
            The pre-processed dataset.
        """
        env.logger.info("Reading raw data file %s.", self.cfg.processing.data_path)

        sep = self.cfg.processing.separator if self.cfg.processing.separator else ','
        data = pd.read_csv(self.cfg.processing.data_path,
                           sep=sep,
                           header=self.cfg.processing.header,
                           engine='python')

        cnt = len(data)
        if self.cfg.processing.threshold and len(data.columns.values) > 2:
            data = data[data[data.columns.values[2]] > float(self.cfg.processing.threshold)]
            if cnt - len(data) > 0:
                env.logger.warning("Thresholded %d ratings.", cnt - len(data))

        env.logger.info("Applying filtering.")
        imin, umin = int(self.cfg.processing.i_min), int(self.cfg.processing.u_min)
        cnt = len(data)
        data = self._filter(data, umin, imin)

        if cnt - len(data) > 0:
            env.logger.warning("Filtered %d ratings.", cnt - len(data))

        return data

    def split(self, data):
        r"""Split the data set.

        The splitting relies on the configurations provided in the data configurations
        :attr:`cfg`. The splitting procedure follows a specific pipeline:

        1. Split the users in training, validation and test sets;
        2. (In case of vertical splitting) Split the validation and test set user ratings \
           in training and test items according to ``test_prop``;
        3. Returns the corresponding :class:`Dataset` object.

        .. warning:: In step (1) there is the possibility that users in the validation or test set \
           have less than 2 ratings making step (2) inconsistent for those users. For this reason, \
           this set of users is simply discarded.

        .. warning:: In step (2) there is the possibility that users in the validation or test set \
           have a number of items which could cause problems in applying the division between \
           training items and test items (e.g., users with 2 ratings and ``test_prop`` = 0.1). \
           In these cases, it is always guaranteed that there is at least one item in the test \
           part of the users.

        Parameters
        ----------
        data : :class:`pandas.DataFrame`
            The dataset to split.

        Returns
        -------
        :class:`Dataset` or :obj:`list` of :class:`Dataset`
            The splitted dataset(s).
        """
        splitted = self._split(data, **self.cfg.splitting)
        if isinstance(splitted, list):
            return [Dataset(*spl) for spl in splitted]
        else:
            return Dataset(*splitted)

    def _split(self,
               data,
               split_type,
               valid_size,
               test_size,
               test_prop,
               sort_by,
               shuffle,
               seed,
               cv=1):
        assert isinstance(cv, int) and cv >= 1
        if split_type == "horizontal":
            if cv == 1:
                return self._horizontal_split(data, valid_size, test_size, sort_by, shuffle, seed)
            else:
                return [self._horizontal_split(data,
                                               valid_size,
                                               test_size,
                                               seed=i * 98765 + seed) for i in range(cv)]
        elif split_type == "vertical":
            if cv == 1:
                return self._vertical_split(data,
                                            valid_size,
                                            test_size,
                                            test_prop,
                                            sort_by,
                                            shuffle,
                                            seed)
            else:
                return [self._vertical_split(data,
                                             valid_size,
                                             test_size,
                                             test_prop,
                                             sort_by,
                                             seed=i * 98765 + seed) for i in range(cv)]
        else:
            raise ValueError("Splitting type must be 'vertical' or 'horizontal'")

    def _get_count(self, data, idx):
        return data[[idx]].groupby(idx, as_index=False).size()

    def _filter(self, data, min_u=5, min_i=0):
        [uhead, ihead] = data.columns.values[:2]
        if min_i > 0:
            icnt = self._get_count(data, ihead)
            data = data[data[ihead].isin(icnt.iloc[:, 0][icnt["size"] >= min_i])]

        if min_u > 0:
            ucnt = self._get_count(data, uhead)
            data = data[data[uhead].isin(ucnt.iloc[:, 0][ucnt["size"] >= min_u])]

        return data

    def _horizontal_split(self, data, valid_size, test_size, sort_by=None, shuffle=True, seed=None):
        assert 0 <= valid_size <= 1, "Invalid validation set size"
        assert 0 < test_size <= 1, "Invalid test set size"

        uhead, ihead = data.columns.values[:2]

        if shuffle and sort_by is None:
            env.logger.info("Shuffling data.")
            if seed is not None:
                np.random.seed(seed)
            data = data.reindex(np.random.permutation(data.index))
        elif sort_by is not None:
            data = data.sort_values(by=[sort_by])

        data_grouped_by_user = data.groupby(uhead)
        env.logger.info("Creating training, validation and test set.")
        tr_list, val_list, te_list = [], [], []
        for _, group in data_grouped_by_user:
            n_items_u = len(group)
            if n_items_u > 2 or (n_items_u == 2 and valid_size == 0):
                if valid_size > 0:
                    uval_sz = max(1, int(n_items_u * valid_size)) if valid_size < 1 else 1
                else:
                    uval_sz = 0
                ute_sz = max(1, int(n_items_u * test_size)) if test_size < 1 else 1
                tr_list.append(group[:-uval_sz - ute_sz])
                if uval_sz > 0:
                    val_list.append(group[-uval_sz - ute_sz:-ute_sz])
                te_list.append(group[-ute_sz:])
            else:
                tr_list.append(group)

        data_tr = pd.concat(tr_list)
        data_val = pd.concat(val_list) if val_list else None
        data_te = pd.concat(te_list)

        unique_iid = pd.unique(data_tr[ihead])
        unique_uid = pd.unique(data_tr[uhead])

        if data_val is not None:
            vcnt = len(data_val)
            data_val = data_val.loc[data_val[ihead].isin(unique_iid)]
            if vcnt - len(data_val) > 0:
                env.logger.warning("Skipped %d ratings in validation set.", vcnt - len(data_val))

        tcnt = len(data_te)
        data_te = data_te.loc[data_te[ihead].isin(unique_iid)]
        if tcnt - len(data_te) > 0:
            env.logger.warning("Skipped %d ratings in test set.", tcnt - len(data_te))

        return data_tr, data_val, data_te, unique_uid, unique_iid

    def _vertical_split(self,
                        data,
                        valid_size,
                        test_size,
                        test_prop=.2,
                        sort_by=None,
                        shuffle=True,
                        seed=None):
        assert valid_size >= 0, "Invalid validation set size"
        assert test_size > 0, "Invalid test set size"
        assert 0 < test_prop <= 1, "Invalid test_prop"

        if seed is not None:
            np.random.seed(seed)
        uhead, ihead = data.columns.values[:2]
        cnt = self._get_count(data, uhead)

        unique_uid = cnt.iloc[:, 0]
        idx_perm = list(range(unique_uid.size))
        if shuffle:
            env.logger.info("Shuffling data.")
            idx_perm = np.random.permutation(unique_uid.size)
            unique_uid = unique_uid[idx_perm]
        else:
            seed = None
        if sort_by:
            env.logger.info("Sorting users' ratings.")
            data = data.sort_values(by=[sort_by])
            seed = None

        n_users = unique_uid.size
        valid_heldout = int(valid_size * n_users) if valid_size < 1 else valid_size
        test_heldout = int(test_size * n_users) if test_size < 1 else test_size

        env.logger.info("Calculating splits.")
        tr_users = unique_uid[:(n_users - valid_heldout - test_heldout)]
        vd_users = unique_uid[(n_users - valid_heldout - test_heldout): (n_users - test_heldout)]
        te_users = unique_uid[(n_users - test_heldout):]

        train_data = data.loc[data[uhead].isin(tr_users)]

        unique_iid = pd.unique(train_data[ihead])

        env.logger.info("Creating validation and test set.")
        val_data = data.loc[data[uhead].isin(vd_users)]
        vcnt = len(val_data)
        val_data = val_data.loc[val_data[ihead].isin(unique_iid)]
        test_data = data.loc[data[uhead].isin(te_users)]
        tcnt = len(test_data)
        test_data = test_data.loc[test_data[ihead].isin(unique_iid)]

        if vcnt - len(val_data) > 0:
            env.logger.warning("Skipped %d ratings in validation set.", vcnt - len(val_data))
        if tcnt - len(test_data) > 0:
            env.logger.warning("Skipped %d ratings in test set.", tcnt - len(test_data))

        vcnt = self._get_count(val_data, uhead)
        tcnt = self._get_count(test_data, uhead)
        val_data = val_data.loc[val_data[uhead].isin(vcnt[vcnt["size"] >= 2].iloc[:, 0])]
        test_data = test_data.loc[test_data[uhead].isin(tcnt[tcnt["size"] >= 2].iloc[:, 0])]

        vcnt_diff = len(vcnt) - len(pd.unique(val_data[uhead]))
        tcnt_diff = len(tcnt) - len(pd.unique(test_data[uhead]))
        if vcnt_diff > 0:
            env.logger.warning("Skipped %d users in validation set.", vcnt_diff)
        if tcnt_diff > 0:
            env.logger.warning("Skipped %d users in test set.", tcnt_diff)

        if valid_size > 0:
            val_data_tr, val_data_te = self._split_train_test(val_data, test_prop, seed)
        else:
            val_data_tr, val_data_te = None, None
        test_data_tr, test_data_te = self._split_train_test(test_data, test_prop, seed)

        return (train_data,
                (val_data_tr, val_data_te),
                (test_data_tr, test_data_te),
                unique_uid,
                unique_iid)

    def _split_train_test(self, data, test_prop, seed=None):
        if seed is not None:
            np.random.seed(seed)
        uhead = data.columns.values[0]
        data_grouped_by_user = data.groupby(uhead)
        tr_list, te_list = [], []
        for _, group in data_grouped_by_user:
            n_items_u = len(group)
            if n_items_u >= 2:
                ute_sz = max(1, int(n_items_u * test_prop)) if test_prop < 1 else 1
                if seed and seed >= 0:
                    idx = np.zeros(n_items_u, dtype='bool')
                    id_sel = np.random.choice(n_items_u, size=ute_sz, replace=False).astype('int64')
                    idx[id_sel] = True
                    tr_list.append(group[np.logical_not(idx)])
                    te_list.append(group[idx])
                else:
                    tr_list.append(group[:-ute_sz])
                    te_list.append(group[-ute_sz:])
            else:
                tr_list.append(group)
        data_tr = pd.concat(tr_list)
        data_te = pd.concat(te_list)
        return data_tr, data_te


class NCRDataProcessing(DataProcessing):
    r"""Class that manages the pre-processing of raw data sets for Neural Collaborative Reasoning (NCR).

    Data sets are expected of being `csv <https://it.wikipedia.org/wiki/Comma-separated_values>`_
    files where each row represents a rating. More details about the allowed format are described
    in :ref:`csv-format` (please, refer to the section dedicated to NCR).

    The pre-processing is performed following the parameters settings defined in the data configuration
    file. Visit :ref:`config-format` for more information (please, refer to the section dedicated to NCR).

    Parameters
    ----------
    data_config : :class:`rectorch.configuration.DataConfig`, :obj:`str`: or :obj:`dict`
        Represents the data pre-processing configurations.
        When ``type(data_config) == str`` is expected to be the path to the data configuration file.
        When ``type(data_config) == dict`` is expected to be the data configuration dictionary.
        In that case a :class:`configuration.DataConfig` object is contextually created.
    numerize : :obj:`bool` [optional]
        Whether the user/item ids must be re-mapped, by default :obj:`True`.

    Raises
    ------
    :class:`TypeError`
        Raises when the type of the input parameter is incorrect.

    Attributes
    ----------
    cfg : :class:`rectorch.configuration.DataConfig`
        The :class:`rectorch.configuration.DataConfig` object containing the pre-processing
        configurations.
    dataset: :class:`pandas.DataFrame`
        The entire raw dataset.
    """
    def __init__(self, data_config, numerize=True):
        super(NCRDataProcessing, self).__init__(data_config)
        sep = self.cfg.processing.separator if self.cfg.processing.separator else ','
        self.dataset = pd.read_csv(self.cfg.processing.data_path,
                                   sep=sep,
                                   header=self.cfg.processing.header,
                                   engine='python')

        if numerize:
            self._prepare_dataset()

    def _prepare_dataset(self):
        """
        It creates the new mapping for user and item ids and create the dataset structure for Neural Collaborative
        Reasoning.
        """
        uhead, ihead = self.dataset.columns.values[:2]
        uids = pd.unique(self.dataset[uhead])
        iids = pd.unique(self.dataset[ihead])
        u2id = dict((uid, i) for (i, uid) in enumerate(uids))
        i2id = dict((iid, i) for (i, iid) in enumerate(iids))
        uid = self.dataset[uhead].apply(lambda x: u2id[x])
        iid = self.dataset[ihead].apply(lambda x: i2id[x])
        dic_data = {'userID': uid, 'itemID': iid, 'rating': pd.to_numeric(self.dataset[self.dataset.columns.values[2]]),
                    'timestamp': self.dataset[self.dataset.columns.values[3]].astype(int)}
        cols = ['userID', 'itemID', 'rating', 'timestamp']
        self.dataset = pd.DataFrame(data=dic_data, columns=cols)

    def process(self):
        r"""It processes the dataset given the pre-processing parameters in the provided configurations :attr:`cfg`.

        It filters the user-item interactions using the ``rating_threshold`` parameter and orders them by
        timestamp field (if ``rating_order`` parameter is set to True). Ratings equal
        to or higher than ``rating_threshold`` are converted to 1 (positive interactions), while ratings lower than
        ``rating_threshold`` are converted to 0 (negative interactions).

        Returns
        ----------
        :class:`pandas.DataFrame`
            The dataset processed according to the pre-processing parameters specified in the configurations :attr:`cfg`.
        """
        # filter ratings by threshold
        proc_dataset = self.dataset.copy()
        proc_dataset['rating'][proc_dataset['rating'] < self.cfg.processing.rating_threshold] = 0
        proc_dataset['rating'][proc_dataset['rating'] >= self.cfg.processing.rating_threshold] = 1

        if self.cfg.processing.rating_order:
            proc_dataset = proc_dataset.sort_values(by=['timestamp', 'userID', 'itemID']).reset_index(drop=True)

        return proc_dataset

    def split(self, data):
        r"""It creates train, validation and test folds as reported in the NCR paper (leave-one-out procedure).

        To split the dataset it uses the splitting configuration expressed in the configurations :attr:`cfg`.
        After the split has been performed a :class:`NCRDataset` is returned.

        Parameters
        ----------
        data : :class:`pandas.DataFrame`
            The processed dataset ordered by timestamp.

        Returns
        ----------
        :class:`NCRDataset`
            The dataset ready to be used for training, validating and testing the NCR model.
        """
        folds = self._leave_one_out_by_time(data, self.cfg.splitting.leave_n, self.cfg.splitting.keep_n)
        folds = self._generate_histories(folds, self.cfg.processing.max_history_length, self.cfg.processing.premise_threshold)
        return NCRDataset(folds[0], folds[1], folds[2], pd.unique(self.dataset['userID']),
                          pd.unique(self.dataset['itemID']), self.dataset)

    def _leave_one_out_by_time(self, data, leave_n=1, keep_n=5):
        """
        It generates train, validation, and test folds of the dataset using the procedure reported in the NCR paper.
        The procedure starts with the dataset ordered by timestamp.
        In particular:
            - the first keep_n positive interactions of each user are put in training set;
            - the last leave_n positive interactions of each user are held out for test set;
            - the second to the last leave_n interactions of each user are held out for validation set.
        :param leave_n: number of items that are left in validation and test set.
        :param keep_n: minimum number of positive interactions to leave in training dataset for each user.
        """

        train_set = []
        # generate training set by looking for the first keep_n POSITIVE interactions
        processed_data = data.copy()
        for uid, group in processed_data.groupby('userID'):  # group by uid
            found, found_idx = 0, -1
            for idx in group.index:
                if group.loc[idx, 'rating'] > 0:
                    found_idx = idx
                    found += 1
                    if found >= keep_n:
                        break
            if found_idx > 0:
                train_set.append(group.loc[:found_idx])
        train_set = pd.concat(train_set)
        # drop the training data info
        processed_data = processed_data.drop(train_set.index)

        # generate test set by looking for the last leave_n POSITIVE interactions
        test_set = []
        for uid, group in processed_data.groupby('userID'):
            found, found_idx = 0, -1
            for idx in reversed(group.index):
                if group.loc[idx, 'rating'] > 0:
                    found_idx = idx
                    found += 1
                    if found >= leave_n:
                        break
            if found_idx > 0:
                test_set.append(group.loc[found_idx:])
        test_set = pd.concat(test_set)
        processed_data = processed_data.drop(test_set.index)

        validation_set = []
        for uid, group in processed_data.groupby('userID'):  #
            found, found_idx = 0, -1
            for idx in reversed(group.index):
                if group.loc[idx, 'rating'] > 0:
                    found_idx = idx
                    found += 1
                    if found >= leave_n:
                        break
            # put all the negative interactions encountered during the search process into validation set
            if found_idx > 0:
                validation_set.append(group.loc[found_idx:])
        validation_set = pd.concat(validation_set)
        processed_data = processed_data.drop(validation_set.index)

        # The remaining data (after removing validation and test) are all in training data
        train_set = pd.concat([train_set, processed_data])
        valid_set, test_set = validation_set.reset_index(drop=True), test_set.reset_index(drop=True)

        return train_set, valid_set, test_set

    def _generate_histories(self, folds, max_hist_length=5, premise_threshold=0):
        """
        Generate history interaction sequence (items at the left side of implication) for each interaction in train,
        validation, and test sets, and appends it to the dataframe.
        In particular, it adds to the dataframe three columns:
            - history column: it contains the items to be put at the left side of implication;
            - history_feedback column: it contains the feedback for the items in the history;
            - history_length column: it contains the length of the history. This is needed for creating batches.
            In particular, we force each batch to have only samples with the same history length.
        :param max_hist_length: the max history length to keep (max number of items at the left side of the
        implication), ==0 value means keeps all.
        :param premise_threshold: it specifies a threshold for filtering logical expressions based on
        the number of premises. Specifically, all the logical expressions with a number of premises equal to or lower
        than premise_threshold will be removed from the dataset. The value should be between 0 (no filter) and
        max_hist_length - 1 (maximum applicable filter). In fact, if premise_threshold is higher than
        max_hist_length - 1, than all the logical expressions will be removed from the dataset.
        For example, if we have logical expressions: a -> b, a ∧ b -> c and a ∧ b ∧ c -> d, and the parameter
        premise_threshold is set to 2, the first two expressions will be removed from the dataset.
        """
        assert premise_threshold < max_hist_length, "premise_threshold cannot be equal to or higher than max_hist_length"
        history_dict = {}  # it contains for each user the list of all the items he has seen
        feedback_dict = {}  # it contains for each user the list of feedbacks he gave to the items he has seen
        for df in folds:
            history = []  # each element of this list is a list containing the history items of a single interaction
            fb = []  # each element of this list is a list containing the feedback for the history items of a
            # single interaction
            hist_len = []  # each element of this list indicates the number of history items of a single interaction
            uids, iids, feedbacks = df['userID'].tolist(), df['itemID'].tolist(), df['rating'].tolist()
            for i, uid in enumerate(uids):
                iid, feedback = iids[i], feedbacks[i]

                if uid not in history_dict:
                    history_dict[uid] = []
                    feedback_dict[uid] = []

                # list containing the history for current interaction
                tmp_his = copy.deepcopy(history_dict[uid]) if max_hist_length == 0 else history_dict[uid][
                                                                                        -max_hist_length:]
                # list containing the feedbacks for the history of current interaction
                fb_his = copy.deepcopy(feedback_dict[uid]) if max_hist_length == 0 else feedback_dict[uid][
                                                                                        -max_hist_length:]

                history.append(tmp_his)
                fb.append(fb_his)
                hist_len.append(len(tmp_his))

                history_dict[uid].append(iid)
                feedback_dict[uid].append(feedback)

            df['history'] = history
            df['history_feedback'] = fb
            df['history_length'] = hist_len

        # filtering logical expressions based on number of premises
        # we remove from the dataset all the logical expressions with a number of premises equal to or lower than
        # premise_threshold
        if premise_threshold != 0:
            folds = [df[df.history_length > premise_threshold] for df in folds]

        return self._clean_data(folds)

    def _clean_data(self, folds):
        """
        It removes all the interactions for which is not possible to construct a logical expression.
        In particular, it removes from train, validation and test sets those interactions (items at the right side of
        implication) that have a negative feedback. In fact, we want to predict only the positive items.
        Then, it removes all the interactions that have an empty history (the left side of implication would be empty).
        """
        folds = [df[df['rating'] > 0].reset_index(drop=True) for df in folds]
        folds = [df[df['history_feedback'].map(len) > 0].reset_index(drop=True) for df in folds]
        return folds


class NCRDataset(Dataset):
    r"""Dataset for training, validating and testing Neural Collaborative Reasoning (NCR).

    For more information about
    this approach, please refer to the official paper: <https://arxiv.org/pdf/2005.08129.pdf>`_. :class:`NCRDataset`
    contains the training, validation, and test set for performing experiments with NCR. This class
    extends from :class:`Dataset` and adds some important features useful for NCR, for example :attr:`user_item_matrix`.

    Parameters
    ----------
    uids : :obj:`list` of :obj:`int`
        See :class:`Dataset`.
    iids : :obj:`list` of :obj:`int`
        See :class:`Dataset`.
    train_set : :class:`pandas.DataFrame`
        The training set data frame.
    valid_set : :class:`pandas.DataFrame`
        The validation set data frame.
    test_set : :class:`pandas.DataFrame`
        The test set data frame.
    full_data : :class:`pandas.DataFrame`
        The entire dataset that has not be split into train, validation, and test sets.

    Attributes
    ----------
    n_users : :obj:`int`
        The number of users.
    n_items : :obj:`int`
        The number of items.
    train_set : :class:`pandas.DataFrame`
        See ``train_set`` parameter.
    valid_set : :class:`pandas.DataFrame`
        See ``valid_set`` parameter.
    test_set : :class:`pandas.DataFrame`
        See ``test_set`` parameter.
    dataset : :class:`pandas.DataFrame`
        See ``full_data`` parameter.
    user_item_matrix : :class:`scipy.sparse.csr_matrix`
        The sparse user-item binary matrix containing the user-item interactions of the entire dataset.
    """

    def __init__(self, train_set, valid_set, test_set, uids, iids, full_data):
        super(NCRDataset, self).__init__(train_set, valid_set, test_set, uids, iids, False)

        self.dataset = full_data

        self.n_users = self.dataset[self.dataset.columns.values[0]].nunique()
        self.n_items = self.dataset[self.dataset.columns.values[1]].nunique()

        self.user_item_matrix = self._compute_sparse_matrix()

    def _compute_sparse_matrix(self):
        """
        It computes the user-item sparse matrix. Every row is a user and every column is an item.
        A 1 in the matrix means that the user liked the item, while a 0 means that the user disliked the item.

        Returns
        -------
        :class:`scipy.sparse.csr_matrix`
            The sparse user-item matrix.
        """
        group = self.dataset.groupby("userID")
        rows, cols = [], []
        values = []
        for i, (_, g) in enumerate(group):
            u = list(g['userID'])[0]  # user id
            items = set(list(g['itemID']))  # items on the history
            rows.extend([u] * len(items))
            cols.extend(list(items))
            values.extend([1] * len(items))
        return csr_matrix((values, (rows, cols)), (self.n_users, self.n_items))
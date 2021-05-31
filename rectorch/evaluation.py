r"""Module containing utility functions to evaluate recommendation engines.
"""
import random
import numpy as np
from scipy.sparse import csr_matrix
from rectorch import env
from rectorch.metrics import Metrics
from rectorch.utils import prepare_for_prediction
import torch

# AUTHORSHIP
__version__ = "0.9.0dev"
__author__ = "Mirko Polato"
__copyright__ = "Copyright 2020, rectorch"
__license__ = "MIT"
__maintainer__ = "Mirko Polato"
__email__ = "mak1788@gmail.com"
__status__ = "Development"
#

__all__ = ['evaluate', 'one_plus_random', 'logic_evaluate']


def evaluate(model, test_sampler, metric_list):
    r"""Evaluate the given model.

    The ``model`` evaluation is performed with all the provided metrics in ``metric_list``.
    The test set is loaded through the provided :class:`rectorch.samplers.Sampler`
    (i.e., ``test_sampler``).

    Parameters
    ----------
    model : :class:`rectorch.models.RecSysModel`
        The model to evaluate.
    test_sampler : :class:`rectorch.samplers.Sampler`
        The test set loader.
    metric_list : :obj:`list` of :obj:`str`
        The list of metrics to compute. Metrics are indicated by strings formed in the
        following way:

        ``matric_name`` @ ``k``

        where ``matric_name`` must correspond to one of the
        method names without the suffix '_at_k', and ``k`` is the corresponding parameter of
        the method and it must be an integer value. For example: ``ndcg@10`` is a valid metric
        name and it corresponds to the method
        :func:`ndcg_at_k <rectorch.metrics.Metrics.ndcg_at_k>` with ``k=10``.

    Returns
    -------
    :obj:`dict` of :obj:`numpy.array`
        Dictionary with the results for each metric in ``metric_list``. Keys are string
        representing the metric, while the value is an array with the value of the metric
        computed on the users.
    """
    if test_sampler.mode == "train":
        test_sampler.test()
        env.logger.warning("Sampler must be in valid or test mode. Forced switch to test mode!")

    results = {m:[] for m in metric_list}
    for _, (data_input, ground_truth) in enumerate(test_sampler):
        data_input, ground_truth = prepare_for_prediction(data_input, ground_truth)
        prediction = model.predict(*data_input)[0].cpu().numpy()
        res = Metrics.compute(prediction, ground_truth, metric_list)
        for m in res:
            results[m].append(res[m])

    for m in results:
        results[m] = np.concatenate(results[m])
    return results


def one_plus_random(model, test_sampler, metric_list, r=1000):
    r"""One plus random evaluation.

    The ``model`` evaluation is performed with all the provided metrics in ``metric_list``.
    The test set is loaded through the provided :class:`rectorch.samplers.Sampler`
    (i.e.,  ``test_sampler``). The evaluation methodology is one-plus-random [OPR]_ that can be
    summarized as follows. For each user, and for each test items, ``r`` random negative items are
    chosen and the metrics are computed w.r.t. to this subset of items (``r`` + 1 items in total).

    Parameters
    ----------
    model : :class:`rectorch.models.RecSysModel`
        The model to evaluate.
    test_sampler : :class:`rectorch.samplers.Sampler`
        The test set loader.
    metric_list : :obj:`list` of :obj:`str`
        The list of metrics to compute. Metrics are indicated by strings formed in the
        following way:

        ``matric_name`` @ ``k``

        where ``matric_name`` must correspond to one of the
        method names without the suffix '_at_k', and ``k`` is the corresponding parameter of
        the method and it must be an integer value. For example: ``ndcg@10`` is a valid metric
        name and it corresponds to the method
        :func:`ndcg_at_k <rectorch.metrics.Metrics.ndcg_at_k>` with ``k=10``.
    r : :obj:`int`
        Number of negative items to consider in the "random" part.

    Returns
    -------
    :obj:`dict` of :obj:`numpy.array`
        Dictionary with the results for each metric in ``metric_list``. Keys are string
        representing the metric, while the value is an array with the value of the metric
        computed on the users.

    References
    ----------
    .. [OPR] Alejandro Bellogin, Pablo Castells, and Ivan Cantador. Precision-oriented Evaluation of
       Recommender Systems: An Algorithmic Comparison. In Proceedings of the Fifth ACM Conference on
       Recommender Systems (RecSys '11). ACM, New York, NY, USA, 333–336, 2011.
       DOI: https://doi.org/10.1145/2043932.2043996
    """
    if test_sampler.mode == "train":
        test_sampler.test()
        env.logger.warning("Sampler must be in valid or test mode. Forced switch to test mode!")

    results = {m:[] for m in metric_list}
    for _, (data_input, ground_truth) in enumerate(test_sampler):
        data_input, ground_truth = prepare_for_prediction(data_input, ground_truth)
        prediction = model.predict(*data_input)[0].cpu().numpy()

        if isinstance(ground_truth, list):
            users = [u for u, iu in enumerate(ground_truth) for _ in range(len(iu))]
            items = [i for iu in ground_truth for i in iu]
            pos = {u: set(ground_truth[u]) for u in users}
        elif isinstance(ground_truth, (np.ndarray, csr_matrix)):
            users, items = ground_truth.nonzero()
            pos = {u : set(list(ground_truth[u].nonzero()[0])) for u in users}
        else:
            raise TypeError("Unrecognized 'ground_truth' type.")
        #elif isinstance(ground_truth, torch.FloatTensor):
        #    users, items = ground_truth.nonzero().t()
        #    pos = {u : set(list(ground_truth[u].nonzero().t()[0])) for u in users}

        rows = []
        tot = set(range(prediction.shape[1]))
        for u, i in zip(users, items):
            #rnd = random.sample(tot - set(list(ground_truth[u].nonzero()[0])), r)
            rnd = random.sample(tot - pos[u], r)
            rows.append(list(prediction[u][[i] + list(rnd)]))

        pred = np.array(rows)
        gt = np.zeros_like(pred)
        gt[:, 0] = 1
        res = Metrics.compute(pred, gt, metric_list)
        for m in res:
            results[m].append(res[m])

    for m in results:
        results[m] = np.concatenate(results[m])
    return results


def logic_evaluate(model, test_sampler, metric_list):
    """Evaluate the given NCR model.

    The ``model`` evaluation is performed with all the provided metrics in ``metric_list``.
    The test set is loaded through the provided :class:`rectorch.samplers.Sampler`
    (i.e., ``test_sampler``). Note that the test loader contains one positive expression
    and 100 negative expressions for each user-item interaction in the test set. The network computes the predictions
    for all these 101 expressions and then the evaluation consists on computing ranking metrics based on the position
    of the target item (positive interactions) in the ranking. The position of an item in the ranking is given by the
    prediction that the network outputs for the logical expression with that item at the right side of implication.
    The prediction is represented as the cosine similarity between the logical expression event vector and the TRUE
    vector. Higher the similarity higher the position of the target item in the ranking.

    Parameters
    ----------
    model : :class:`rectorch.models.RecSysModel`
        The model to evaluate.
    test_sampler : :class:`rectorch.samplers.Sampler`
        The test set loader.
    metric_list : :obj:`list` of :obj:`str`
        The list of metrics to compute. Metrics are indicated by strings formed in the
        following way:

        ``metric_name`` @ ``k``

        where ``metric_name`` must correspond to one of the
        method names without the suffix '_at_k', and ``k`` is the corresponding parameter of
        the method and it must be an integer value. For example: ``ndcg@10`` is a valid metric
        name and it corresponds to the method
        :func:`ndcg_at_k <rectorch.metrics.Metrics.ndcg_at_k>` with ``k=10``.

    Returns
    -------
    :obj:`dict` of :obj:`numpy.array`
        Dictionary with the results for each metric in ``metric_list``. Keys are string
        representing the metrics, while values are arrays with the values of the metrics
        computed on the test interactions.
    """
    if test_sampler.mode == "train":
        test_sampler.test()
        env.logger.warning("Sampler must be in valid or test mode. Forced switch to test mode!")

    results = {m:[] for m in metric_list}
    for batch_idx, batch_data in enumerate(test_sampler):
        positive_pred, negative_pred = model.predict(batch_data)
        # we concatenate the positive prediction to the negative predictions
        # in each row of the final tensor we will have the positive prediction in the first column
        # and the 100 negative predictions in the last 100 columns
        positive_pred = positive_pred.view(positive_pred.size(0), 1)
        pred_scores = torch.cat((positive_pred, negative_pred), dim=1)
        # now, we need to construct the ground truth tensor
        ground_truth = np.zeros(pred_scores.size())
        ground_truth[:, 0] = 1  # the positive item is always in the first column of pred_scores, as we said before
        pred_scores = pred_scores.cpu().numpy()
        res = Metrics.compute(pred_scores, ground_truth, metric_list)
        for m in res:
            results[m].append(res[m])

    for m in results:
        results[m] = np.concatenate(results[m])
    return results

r"""A Generic Collaborative Filtering Framework based on Generative Adversarial Networks.

This recommender system is based on GAN where the (conditioned) generator aims at generating
the ratings of users while the discriminiator tries to discriminate between genuine users
profile and generated ones.
The two networks try to optimize the followng loss functions:

- Discriminator (D):
    :math:`J^{D}=-\sum_{u}\left(\log D(\mathbf{r}_{u} | \
    \mathbf{c}_{u})+\log (1-D(\hat{\mathbf{r}}_{u}\
    \odot(\mathbf{e}_{u}+\mathbf{k}_{u}) | \mathbf{c}_{u}))\right)`
- Generator (G):
    :math:`J^{G}=\sum_{u}\left(\log (1-D(\hat{\mathbf{r}}_{u}\
    \odot(\mathbf{e}_{u}+\mathbf{k}_{u}) | \mathbf{c}_{u}))+\
    \alpha \cdot \sum_{j}(r_{uj}-\hat{r}_{uj})^{2}\right)`

where :math:`\mathbf{c}_u` is the condition vector, :math:`\mathbf{k}_u` is an n-dimensional
indicator vector such that :math:`k_{uj} = 1` iff
*j* is a negative sampled item, :math:`\hat{\mathbf{r}}_u` is a fake user profile, and
:math:`\mathbf{e}_u` is the masking vector to remove the non-rated items.
For more details please refer to the original paper [CFGAN]_.

References
----------
.. [CFGAN] Dong-Kyu Chae, Jin-Soo Kang, Sang-Wook Kim, and Jung-Tae Lee. 2018.
   CFGAN: A Generic Collaborative Filtering Framework based on Generative Adversarial Networks.
   In Proceedings of the 27th ACM International Conference on Information and Knowledge
   Management (CIKM '18). Association for Computing Machinery, New York, NY, USA, 137–146.
   DOI: https://doi.org/10.1145/3269206.3271743
"""
import time
import random
from importlib import import_module
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.init import normal_ as normal_init
from rectorch import env, set_seed
from rectorch.samplers import Sampler
from rectorch.models.nn import NeuralNet, TorchNNTrainer, NeuralModel
from rectorch.evaluation import logic_evaluate
from rectorch.validation import ValidFunc
from rectorch.utils import init_optimizer

# AUTHORSHIP
__version__ = "0.9.0dev"
__author__ = "Mirko Polato"
__copyright__ = "Copyright 2020, rectorch"
__license__ = "MIT"
__maintainer__ = "Mirko Polato"
__email__ = "mak1788@gmail.com"
__status__ = "Development"
#

__all__ = ["NCR_net", "NCR_Sampler", "NCR_trainer", "NCR"]


class NCR_net(NeuralNet):
    r"""Generator network of the CFGAN model.

    The generator newtork of CFGAN is a simple Multi Layer perceptron. Each internal layer is
    fully connected and ReLU activated. The output layer insted has a sigmoid as activation
    funciton. See [CFGAN]_ for a full description.

    Parameters
    ----------
    layers_dim : :obj:`list` of :obj:`int`
        The dimension of the layers of the network ordered from the input to the output.

    Attributes
    ----------
    layers_dim : :obj:`list` of :obj:`int`
        See the :attr:`layers_dim` parameter.
    input_dim : :obj:`int`
        The dimension of the output of the generator, i.e., the input of the discriminator.
    latent_dim : :obj:`int`
        The dimension of the latent space, i.e., the dimension of the input of the generator.

    References
    ----------
    .. [CFGAN] Dong-Kyu Chae, Jin-Soo Kang, Sang-Wook Kim, and Jung-Tae Lee. 2018.
       CFGAN: A Generic Collaborative Filtering Framework based on Generative Adversarial Networks.
       In Proceedings of the 27th ACM International Conference on Information and Knowledge
       Management (CIKM ’18). Association for Computing Machinery, New York, NY, USA, 137–146.
       DOI: https://doi.org/10.1145/3269206.3271743
    """
    def __init__(self, n_users, n_items, emb_size=64, dropout=0.0, remove_double_not=False):
        super(NCR_net, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.emb_size = emb_size
        self.dropout = dropout
        # initialization of user and item embeddings
        self.item_embeddings = torch.nn.Embedding(self.n_items, self.emb_size)
        self.user_embeddings = torch.nn.Embedding(self.n_users, self.emb_size)
        # this is the true anchor vector that is fixed during the training of the model (for this reason it has the
        # requires_grad parameter af False)
        self.true_vector = torch.nn.Parameter(torch.from_numpy(
            np.random.uniform(0, 0.1, size=self.emb_size).astype(np.float32)),
            requires_grad=False)  # gradient is false to disable the training of the vector
        # first layer of NOT network
        self.not_layer_1 = torch.nn.Linear(self.emb_size, self.emb_size)
        # second layer of NOT network (this network has two layers with the same number of neurons)
        self.not_layer_2 = torch.nn.Linear(self.emb_size, self.emb_size)
        # first layer of OR network: it takes two embeddings, so the input size is 2 * emb_size
        self.or_layer_1 = torch.nn.Linear(2 * self.emb_size, self.emb_size)
        # second layer of OR network
        self.or_layer_2 = torch.nn.Linear(self.emb_size, self.emb_size)
        # first layer of AND network (this network is not directly used, it is used only for the logical regularizers)
        self.and_layer_1 = torch.nn.Linear(2 * self.emb_size, self.emb_size)
        # second layer of AND network
        self.and_layer_2 = torch.nn.Linear(self.emb_size, self.emb_size)
        # first layer of encoder: it converts a pair of user-item vectors in an event vector (refer to the paper)
        self.encoder_layer_1 = torch.nn.Linear(2 * self.emb_size, self.emb_size)
        # second layer of encoder
        self.encoder_layer_2 = torch.nn.Linear(self.emb_size, self.emb_size)
        # dropout layer
        self.dropout_layer = torch.nn.Dropout(self.dropout)
        # initialize the weights of the network
        self.init_weights()
        self.remove_double_not = remove_double_not


    def logic_not(self, vector):
        """
        This represents the NOT neural module. It takes in input an event vector and returns a new event vector that
        is the negation of the input event vector.
        :param vector: the input event vector.
        :return: the event vector that is the negation of the input event vector.
        """
        # ReLU is the activation function selected in the paper
        vector = F.relu(self.not_layer_1(vector))
        if self.training:
            vector = self.dropout_layer(vector)
        out = self.not_layer_2(vector)
        return out


    def logic_or(self, vector1, vector2, dim=1):
        """
        This represents the OR neural module. It takes in input two event vectors and returns a new event vector that is
        the logical disjunction of the two input event vectors.
        :param vector1: the first input event vector.
        :param vector2: the second input event vector.
        :param dim: the dimension for the concatenation of the two input event vectors.
        :return: the event vector that is the logical disjunction of the two input event vectors.
        """
        vector = torch.cat((vector1, vector2), dim)
        vector = F.relu(self.or_layer_1(vector))
        if self.training:
            vector = self.dropout_layer(vector)
        out = self.or_layer_2(vector)
        return out


    def logic_and(self, vector1, vector2, dim=1):
        """
        This represents the AND neural module. It takes in input two event vectors and returns a new event vector that
        is the logical conjunction of the two input event vectors.
        :param vector1: the first input event vector.
        :param vector2: the second input event vector.
        :param dim: the dimension for the concatenation of the two input event vectors.
        :return: the event vector that is the logical conjunction of the two input event vectors.
        """
        vector = torch.cat((vector1, vector2), dim)
        vector = F.relu(self.and_layer_1(vector))
        if self.training:
            vector = self.dropout_layer(vector)
        out = self.and_layer_2(vector)
        return out


    def encoder(self, ui_vector):
        """
        This represents the encoder network of the paper. It takes in input the concatenation of two embeddings, a user
        embedding and an item embedding respectively, and it converts it into an event vector. The event vector is an
        embedding that captures the relationship between a user and an item.
        :param ui_vector: the embedding that is the concatenation of a user embedding with an item embedding.
        :return: the event vector that represents the user-item pair in input.
        """
        event_vector = F.relu(self.encoder_layer_1(ui_vector))
        if self.training:
            event_vector = self.dropout_layer(event_vector)
        event_vector = self.encoder_layer_2(event_vector)
        return event_vector


    def forward(self, batch_data):
        """
        This is the function that performs the forward phase of the neural network. In this particular network, the
        forward phase is really complex. First of all, each element in the batch represents a user-item interaction and
        comes with the following information:
            - user id: this is the id of the user.
            - item id: this is the id of the item. This is the positive item we want to predict, namely the item at the
            right side of the implication of the logical expression.
            - history: this is a numpy array that forms the history sequence of the user-item interaction. It could be
            an array of up to 5 elements. The history contains the items that have to be placed at the left side of the
            implication of the logical expression.
            - feedback of history: this is a numpy array of the same length of history. It contains only 1/0 and
            specifies which items in the history are negative items (that have to be negated with the NOT module) and
            which items are positive items. 1 means positive, while 0 means negative.
            - negative item: during training this is the negative item to build the negative logical expression for
            the pair-wise learning. During validation, instead, we have 100 negative items to build 100 negative
            expressions.

        The forward function takes as input these information for every user-item interaction in the batch, it
        constructs the positive logical expression (history -> positive item ID) and negative logical expression
        (history -> negative item ID) based on these information and finally it computes the similarity of the two
        logical expressions with the TRUE vector. An high similarity means that the logical expression is true and the
        target item could be recommended, while a low similarity means that the logical expression is false and the
        target item should not be recommended. During training, we want the logical expressions based on positive items
        to be evaluated to true, while the logical expressions based on negative items to be evaluated to false.
        Finally, this function adds all the intermediate event vectors that it obtains while constructing the
        logical expressions to a list. These intermediate event vectors are then used by the model for performing the
        logical regularization, that is needed to assure that each module learns the correct logical operator.
        :param batch_data: it contains the user-item interactions and the information to build the positive and
        negative logical expressions.
        :return positive_predictions: a torch tensor containing the similarities of the positive logical expressions
        with the TRUE vector.
        :return negative_predictions: a torch tensor containing the similarities of the negative logical expressions
        with the TRUE vector. Note that during validation we have 100 similarities for each user-item interaction in
        input since we build 100 negative logical expressions.
        :return constraints: a torch tensor of intermediate event vectors used by the model for performing logical
        regularization.
        """
        user_ids, item_ids, histories, history_feedbacks, neg_item_ids = batch_data

        # here, we select the user and item (also negative) embeddings given user, item and negative item ids
        user_embs = self.user_embeddings(user_ids)
        item_embs = self.item_embeddings(item_ids)
        neg_item_embs = self.item_embeddings(neg_item_ids)

        # here, we concatenate the user embeddings with item embeddings to get event embeddings using encoder
        # note that these are the event embeddings of the events at the right side of the implication
        right_side_events = self.encoder(torch.cat((user_embs, item_embs), dim=1))  # positive event vectors at the
        # right side of implication. These are used in the positive logical expressions.

        # in validation we have 100 negative item embeddings for each expression, while in training only 1
        # in order to make this function flexible and usable in both situations, we need to expand the user embeddings
        # in a size compatible with the negative item embeddings
        exp_user_embs = user_embs.view(user_embs.size(0), 1, user_embs.size(1))
        exp_user_embs = exp_user_embs.expand(user_embs.size(0), neg_item_embs.size(1),
                                                             user_embs.size(1))
        right_side_neg_events = self.encoder(torch.cat((exp_user_embs, neg_item_embs), dim=2))  # negative event
        # vectors at the right side of implication. These are used in the negative logical expressions.

        # now, we need the event vectors for the items at the left side of the logical expression
        # expand user embeddings to prepare for concatenating with history item embeddings
        left_side_events = user_embs.view(user_embs.size(0), 1, user_embs.size(1))
        left_side_events = left_side_events.expand(user_embs.size(0), histories.size(1), user_embs.size(1))

        # here, we get the item embeddings for the items in the histories
        history_item_embs = self.item_embeddings(histories)

        # concatenate user embeddings with history item embeddings to get left side event vectors using encoder
        left_side_events = self.encoder(torch.cat((left_side_events, history_item_embs), dim=2))

        # here, we perform the negation of the event embeddings at the left side of the expression
        left_side_neg_events = self.logic_not(left_side_events)

        # we begin to construct the constrains list containing all the intermediate event vectors used in the logical
        # regularization
        constraints = list([left_side_events])
        constraints.append(left_side_neg_events)

        # here, we take the correct (negated or not depending on the feedbacks of the users) event vectors for the
        # logic expression
        # from this instruction it begins the construction of the logical expression through the network
        # here, we expand the feedback_history tensor in order to make it compatible with left_side_events tensor, so
        # that we can perform element-wise multiplication to get the right left side event vectors
        feedback_tensor = history_feedbacks.view(history_feedbacks.size(0), history_feedbacks.size(1), 1)
        feedback_tensor = feedback_tensor.expand(history_feedbacks.size(0), history_feedbacks.size(1), self.emb_size)

        # if we do not want the double negations, we flip the feedback vector and we do not compute the NOTs in the
        # intermediate stages of the network. By doing so, we obtain a logically equivalent expression avoiding
        # the computation of double negations.
        if self.remove_double_not:
            left_side_events = (1 - feedback_tensor) * left_side_events + feedback_tensor * left_side_neg_events
        else:
            left_side_events = feedback_tensor * left_side_events + (1 - feedback_tensor) * left_side_neg_events

        #constraints = list([left_side_events])

        # now, we have the event vectors for the items in the history, we only need to build the logical expression
        # for building the logical expression we need to negate the events in the history and perform an OR operation
        # between them
        # then, we need to do (OR between negated events of the history) OR (event at the right side of implication)

        # here, we negate the events in the history only if we want to compute double negations
        if not self.remove_double_not:
            left_side_events = self.logic_not(left_side_events)
        #constraints.append(left_side_events)

        # now, we perform the logical OR between these negated left side events to build the event that is the logical
        # OR of all these events
        tmp_vector = left_side_events[:, 0]  # we take the first event of history

        shuffled_history_idx = list(range(1, histories.size(1)))  # this is needed to permute the order of the operands
        # in the OR operator at every batch
        np.random.shuffle(shuffled_history_idx)
        for i in shuffled_history_idx:
            tmp_vector = self.logic_or(tmp_vector, left_side_events[:, i])
            constraints.append(tmp_vector.view(histories.size(0), -1, self.emb_size))  # this is done to have all the
            # tensors in the constraint list of the same size
        left_side_events = tmp_vector

        constraints.append(right_side_events.view(histories.size(0), -1, self.emb_size))
        constraints.append(right_side_neg_events)  # this has already the correct shape, so it is not necessary to
        # perform a view

        # these are the event vectors of the entire original logical expressions
        expression_events = self.logic_or(left_side_events, right_side_events)
        constraints.append(expression_events.view(histories.size(0), -1, self.emb_size))

        # here, we have the result of the OR operation at the left side of the implication of the expressions in the
        # batch and we need to perform an OR between each one of these left side expressions and their corresponding
        # 1 or 100 negative right side events
        # so, for each expression we have to perform 1/100 OR operations
        # we need to reshape the results of these OR at the left side of the expressions in order to perform the OR
        # of each one with its corresponding 1/100 negative interactions
        exp_left_side_events = left_side_events.view(left_side_events.size(0), 1, left_side_events.size(1))
        exp_left_side_events = exp_left_side_events.expand(left_side_events.size(0), right_side_neg_events.size(1),
                                                           left_side_events.size(1))
        expression_neg_events = self.logic_or(exp_left_side_events, right_side_neg_events, dim=2)
        constraints.append(expression_neg_events)

        # why times 10? In order to have a sigmoid output between 0 and 1
        # these are the similarities between the positive logical expressions and the TRUE vector
        positive_predictions = F.cosine_similarity(expression_events, self.true_vector.view([1, -1])) * 10  # here the view is
        # used to transpose the true column vector in a row vector
        # these are the similarities between the negative logical expressions and the TRUE vector
        # we need to reshape the tensor containing the negative expressions in such a way to be able to compute the
        # cosine similarity of each expression with the TRUE vector
        reshaped_expression_neg_events = expression_neg_events.reshape(expression_neg_events.size(0) *
                                                        expression_neg_events.size(1), expression_neg_events.size(2))
        negative_predictions = F.cosine_similarity(reshaped_expression_neg_events, self.true_vector.view([1, -1])) * 10
        negative_predictions = negative_predictions.reshape(expression_neg_events.size(0),
                                                            expression_neg_events.size(1))

        # we convert the constraints list in tensor in order to be able compute the loss
        constraints = torch.cat(constraints, dim=1)

        # here, in order to make computation easier, we remove one dimension since it is not needed
        constraints = constraints.view(constraints.size(0) * constraints.size(1), constraints.size(2))

        return positive_predictions, negative_predictions, constraints

    def init_weights(self):
        r"""Initialize the weights of the network.

        Weights are initialized with the :py:func:`torch.nn.init.xavier_uniform_` initializer,
        while biases are initalized with the :py:func:`torch.nn.init.normal_` initializer.
        """
        """
                It initializes all the weights of the neural architecture as reported in the paper.
                """
        # not
        normal_init(self.not_layer_1.weight, mean=0.0, std=0.01)
        normal_init(self.not_layer_1.bias, mean=0.0, std=0.01)
        normal_init(self.not_layer_2.weight, mean=0.0, std=0.01)
        normal_init(self.not_layer_2.bias, mean=0.0, std=0.01)
        # or
        normal_init(self.or_layer_1.weight, mean=0.0, std=0.01)
        normal_init(self.or_layer_1.bias, mean=0.0, std=0.01)
        normal_init(self.or_layer_2.weight, mean=0.0, std=0.01)
        normal_init(self.or_layer_2.bias, mean=0.0, std=0.01)
        # and
        normal_init(self.and_layer_1.weight, mean=0.0, std=0.01)
        normal_init(self.and_layer_1.bias, mean=0.0, std=0.01)
        normal_init(self.and_layer_2.weight, mean=0.0, std=0.01)
        normal_init(self.and_layer_2.bias, mean=0.0, std=0.01)
        # encoder
        normal_init(self.encoder_layer_1.weight, mean=0.0, std=0.01)
        normal_init(self.encoder_layer_1.bias, mean=0.0, std=0.01)
        normal_init(self.encoder_layer_2.weight, mean=0.0, std=0.01)
        normal_init(self.encoder_layer_2.bias, mean=0.0, std=0.01)
        # embeddings
        normal_init(self.user_embeddings.weight, mean=0.0, std=0.01)
        normal_init(self.item_embeddings.weight, mean=0.0, std=0.01)


    def get_state(self):
        state = {
            "name" : self.__class__.__name__,
            "state" : self.state_dict(),
            "params" : {
                "n_users" : self.n_users,
                "n_items" : self.n_items,
                "emb_size" : self.emb_size,
                "dropout" : self.dropout,
                "remove_double_not" : self.remove_double_not
            }
        }
        return state


class NCR_Sampler(Sampler):
    """This is a standard sampler that returns batches without any particular constraint.
    Bathes are randomly returned with the defined dimension (i.e., ``batch_size``). If ``shuffle``
    is set to ``False`` then the sampler returns batches with the same order as in the original
    dataset.
    Parameters/Attributes
    ----------
    data : the dataset fold for which the batches have to be created.
    n_neg_samples: number of negative samples that have to be generated for each interaction (during training should be
    one, while in validation and test should be 100 (these are the specs reported in the paper))
    user_item_matrix: this is a scipy sparse matrix containing the user-item interactions in the dataset. This is needed
    for the sampling of negative items.
    batch_size : the size of the batches, by default 1.
    shuffle : whether the data set must by randomly shuffled before creating the batches, by default ``True``
    device: the device where the torch tensors have to be put
    """
    def __init__(self,
                 data,
                 mode='train',
                 n_neg_samples_t=1,
                 n_neg_samples_vt=100,
                 batch_size=128,
                 shuffle=True):
        super(NCR_Sampler, self).__init__(data, mode, batch_size)
        self.dataset = data
        self.n_neg_samples_t = n_neg_samples_t
        self.n_neg_samples_vt = n_neg_samples_vt
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._set_mode(mode)


    def _set_mode(self, mode="train", batch_size=None):
        assert mode in ["train", "valid", "test"], "Invalid sampler's mode."
        self.mode = mode

        if self.mode == "train":
            self.data = self.dataset.train_set
            self.n_neg_samples = self.n_neg_samples_t
        elif self.mode == "valid":
            self.data = self.dataset.valid_set
            self.n_neg_samples = self.n_neg_samples_vt
        else:
            self.data = self.dataset.test_set
            self.n_neg_samples = self.n_neg_samples_vt

        if batch_size is not None:
            self.batch_size = batch_size

    def __len__(self):
        # here, it is not sufficient to return int(np.ceil(self.data.shape[0] / self.batch_size)) since we have
        # grouped the dataset into small datasets each of one correspond to a history length
        dataset_size = 0
        length = self.data.groupby("history_length")  # histories could be of different lengths, so we need to group
        # histories of the same length in the same batch
        for i, (_, l) in enumerate(length):
            dataset_size += int(np.ceil(l.shape[0] / self.batch_size))
        return dataset_size

    def __iter__(self):
        # for each epoch, each positive interaction has a random sampled negative interaction for the pair-wise learning
        # instead, in validation, each positive interaction has 100 random sampled negative interactions for the
        # computation of the metrics
        length = self.data.groupby("history_length")  # histories could be of different lengths, so we need to group
        # histories of the same length in the same batch

        for i, (_, l) in enumerate(length):
            # get numpy arrays of the dataframe fields that we need to train the model
            group_users = np.array(list(l['userID']))
            group_items = np.array(list(l['itemID']))
            group_histories = np.array(list(l['history']))
            group_feedbacks = np.array(list(l['history_feedback']))

            n = group_users.shape[0]
            idxlist = list(range(n))
            # every small dataset based on history length is shuffled before preparing batches
            if self.shuffle and self.mode == 'train':
                np.random.shuffle(idxlist)

            for _, start_idx in enumerate(range(0, n, self.batch_size)):
                end_idx = min(start_idx + self.batch_size, n)
                batch_users = torch.from_numpy(group_users[idxlist[start_idx:end_idx]])
                batch_items = torch.from_numpy(group_items[idxlist[start_idx:end_idx]])
                batch_histories = torch.from_numpy(group_histories[idxlist[start_idx:end_idx]])
                batch_feedbacks = torch.from_numpy(group_feedbacks[idxlist[start_idx:end_idx]])

                # here, we generate negative items for each interaction in the batch
                batch_user_item_matrix = self.dataset.user_item_matrix[batch_users].toarray()  # this is the portion of the
                # user-item matrix for the users in the batch
                batch_user_unseen_items = 1 - batch_user_item_matrix  # this matrix contains the items that each user
                # in the batch has never seen
                negative_items = []  # this list contains a list of negative items for each interaction in the batch
                for u in range(batch_users.size(0)):
                    u_unseen_items = batch_user_unseen_items[u].nonzero()[0]  # items never seen by the user
                    # here, we generate n_neg_samples indexes and use them to take n_neg_samples random items from
                    # the list of the items that the user has never seen
                    rnd_negatives = u_unseen_items[random.sample(range(u_unseen_items.shape[0]), self.n_neg_samples)]
                    # we append the list to negative_items
                    negative_items.append(rnd_negatives)
                batch_negative_items = torch.tensor(negative_items)

                yield batch_users.to(env.device), batch_items.to(env.device), batch_histories.to(env.device), \
                      batch_feedbacks.to(env.device), batch_negative_items.to(env.device)


class NCR_trainer(TorchNNTrainer):
    r"""Trainer class for the CFGAN model.

    Parameters
    ----------
    generator : :class:`torch.nn.Module`
        The generator neural network. The expected architecture is
        :math:`[m, l_1, \dots, l_h, m]` where *m* is the number of items :math:`l_i, i \in [1,h]`
        is the number of nodes of the hidden layer *i*.
    discriminator : :class:`torch.nn.Module`
        The discriminator neural network. The expected architecture is
        :math:`[2m, l_1, \dots, l_h, 1]` where *m* is the number of items :math:`l_i, i \in [1,h]`
        is the number of nodes of the hidden layer *i*.
    alpha : :obj:`float` [optional]
        The ZR-coefficient, by default 0.1.
    s_pm : :obj:`float` [optional]
        The sampling parameter for the partial-masking approach, by default 0.7.
    s_zr : :obj:`float` [optional]
        The sampling parameter for the zero-reconstruction regularization, by default 0.5.
    opt_conf : :obj:`dict` [optional]
        The optimizer configuration dictionary, by default :obj:`None`.

    Attributes
    ----------
    generator : :class:`torch.nn.Module`
        See ``generator`` parameter.
    discriminator : :class:`torch.nn.Module`
        See ``discriminator`` parameter.
    alpha : :obj:`float`
        See ``alpha`` paramater.
    s_pm : :obj:`float`
        See ``s_pm`` paramater.
    s_zr : :obj:`float`
        See ``s_zr`` paramater.
    opt_g : :class:`torch.optim.Optimizer`
        Optimizer used for performing the training of the generator.
    opt_d : :class:`torch.optim.Optimizer`
        Optimizer used for performing the training of the discriminator.
    """
    def __init__(self,
                 ncr_net,
                 logic_reg_weight=0.01,
                 device=None,
                 opt_conf=None):
        super(NCR_trainer, self).__init__(ncr_net, device, opt_conf)
        self.ncr_net = self.network
        self.reg_weight = logic_reg_weight
        self.optimizer = init_optimizer(self.ncr_net.parameters(), opt_conf)


    def reg_loss(self, constraints):
        """
        It computes the regularization part of the loss function.
        :param constraints: see loss_function()
        :return: the regularization loss for the batch intermediate event vectors given in input.
        """
        false_vector = self.ncr_net.logic_not(self.ncr_net.true_vector)  # we compute the representation
        # for the FALSE vector

        # here, we need to implement the logical regularizers for the logical regularization

        # here, we implement the logical regularizers for the NOT operator

        # minimizing 1 - similarity means maximizing the cosine similarity between the two event vectors in input
        # minimizing 1 + similarity means minimizing the cosine similarity between the two event vectors in input

        # here, we maximize the similarity between not not true and true
        r_not_not_true = (1 - F.cosine_similarity(
            self.ncr_net.logic_not(self.ncr_net.logic_not(self.ncr_net.true_vector)), self.ncr_net.true_vector, dim=0))

        # here, we maximize the similarity between not true and false
        # r_not_true = (1 - F.cosine_similarity(self.ncr_net.logic_not(self.ncr_net.true_vector), false_vector, dim=0))

        # here, we maximize the similarity between not not x and x
        r_not_not_self = \
            (1 - F.cosine_similarity(self.ncr_net.logic_not(self.ncr_net.logic_not(constraints)), constraints)).mean()

        # here, we minimize the similarity between not x and x
        r_not_self = (1 + F.cosine_similarity(self.ncr_net.logic_not(constraints), constraints)).mean()

        # here, we minimize the similarity between not not x and not x
        r_not_not_not = \
            (1 + F.cosine_similarity(self.ncr_net.logic_not(self.ncr_net.logic_not(constraints)),
                                     self.ncr_net.logic_not(constraints))).mean()

        # here, we implement the logical regularizers for the OR operator

        # here, we maximize the similarity between x OR True and True
        r_or_true = (1 - F.cosine_similarity(
            self.ncr_net.logic_or(constraints, self.ncr_net.true_vector.expand_as(constraints)),
            self.ncr_net.true_vector.expand_as(constraints))).mean()

        # here, we maximize the similarity between x OR False and x
        r_or_false = (1 - F.cosine_similarity(
            self.ncr_net.logic_or(constraints, false_vector.expand_as(constraints)), constraints)).mean()

        # here, we maximize the similarity between x OR x and x
        r_or_self = (1 - F.cosine_similarity(self.ncr_net.logic_or(constraints, constraints), constraints)).mean()

        # here, we maximize the similarity between x OR not x and True
        r_or_not_self = (1 - F.cosine_similarity(
            self.ncr_net.logic_or(constraints, self.ncr_net.logic_not(constraints)),
            self.ncr_net.true_vector.expand_as(constraints))).mean()

        # same rule as before, but we flipped operands
        r_or_not_self_inverse = (1 - F.cosine_similarity(
            self.ncr_net.logic_or(self.ncr_net.logic_not(constraints), constraints),
            self.ncr_net.true_vector.expand_as(constraints))).mean()

        # here, we implement the logical regularizers for the AND operator

        # here, we maximize the similarity between x AND True and x
        r_and_true = (1 - F.cosine_similarity(
            self.ncr_net.logic_and(constraints, self.ncr_net.true_vector.expand_as(constraints)), constraints)).mean()

        # here, we maximize the similarity between x AND False and False
        r_and_false = (1 - F.cosine_similarity(
            self.ncr_net.logic_and(constraints, false_vector.expand_as(constraints)),
            false_vector.expand_as(constraints))).mean()

        # here, we maximize the similarity between x AND x and x
        r_and_self = (1 - F.cosine_similarity(self.ncr_net.logic_and(constraints, constraints), constraints)).mean()

        # here, we maximize the similarity between x AND not x and False
        r_and_not_self = (1 - F.cosine_similarity(
            self.ncr_net.logic_and(constraints, self.ncr_net.logic_not(constraints)),
            false_vector.expand_as(constraints))).mean()

        # same rule as before, but we flipped operands
        r_and_not_self_inverse = (1 - F.cosine_similarity(
            self.ncr_net.logic_and(self.ncr_net.logic_not(constraints), constraints),
            false_vector.expand_as(constraints))).mean()

        # True/False rule

        # here, we minimize the similarity between True and False
        true_false = 1 + F.cosine_similarity(self.ncr_net.true_vector, false_vector, dim=0)

        r_loss = r_not_not_true + r_not_not_self + r_not_self + r_not_not_not + \
                 r_or_true + r_or_false + r_or_self + r_or_not_self + r_or_not_self_inverse + true_false + \
                 r_and_true + r_and_false + r_and_self + r_and_not_self + r_and_not_self_inverse

        return r_loss


    def loss_function(self, positive_preds, negative_preds, constraints):
        """
        This method computes the loss function for a single batch. It takes as inputs the predictions for positive
        and negative logical expressions and a tensor containing the intermediate event vectors obtained while building
        the logical expressions of the batch. The loss is computed as reported in the paper.
        :param positive_preds: predictions for positive logical expressions.
        :param negative_preds: predictions for negative logical expressions.
        :param constraints: tensor containing the intermediate event vectors obtained while building the logical
        expressions of the batch.
        :return the partial loss function for the given batch.
        """
        # here, we implement the recommendation pair-wise loss as in the paper
        # since we need to compute the differences between the positive predictions and negative predictions
        # we need to change the size of positive predictions in order to be of the same size of negative predictions
        # this is required because we could have more than one negative expression for each positive expression
        positive_preds = positive_preds.view(positive_preds.size(0), 1)
        positive_preds = positive_preds.expand(positive_preds.size(0), negative_preds.size(1))
        loss = -(positive_preds - negative_preds).sigmoid().log().sum()  # this is the formula in the paper

        # here, we compute the regularization loss
        r_loss = self.reg_loss(constraints)

        return loss + self.reg_weight * r_loss


    def train_epoch(self, epoch, data_sampler, verbose):
        """
        This method performs the training of a single epoch.
        :param epoch: id of the epoch.
        :param train_loader: the DataLoader that loads the training set.
        :param verbose: see train() method.
        """
        data_sampler.train()
        self.ncr_net.train()  # set the network in train mode
        train_loss = 0
        partial_loss = 0
        epoch_start_time = time.time()
        start_time = time.time()
        log_delay = max(10, len(data_sampler) // 10 ** verbose)

        for batch_idx, batch_data in enumerate(data_sampler):
            partial_loss += self.train_batch(batch_data)
            if (batch_idx + 1) % log_delay == 0:
                elapsed = time.time() - start_time
                env.logger.info('| epoch %d | %d/%d batches | ms/batch %.2f | loss %.2f |',
                            epoch, (batch_idx + 1), len(data_sampler),
                            elapsed * 1000 / log_delay,
                            partial_loss / log_delay)
                train_loss += partial_loss
                partial_loss = 0.0
                start_time = time.time()
        total_loss = (train_loss + partial_loss) / len(data_sampler)
        time_diff = time.time() - epoch_start_time
        env.logger.info("| epoch %d | loss %.4f | total time: %.2fs |", epoch, total_loss, time_diff)


    def train_batch(self, batch_data):
        """
        This method performs the training of a single batch.
        Parameters
        ----------
        :param batch_data: this is the batch on which we have to train on.
        :return the partial loss computed on the given batch.
        """
        self.optimizer.zero_grad()
        positive_preds, negative_preds, constraints = self.ncr_net(batch_data)
        loss = self.loss_function(positive_preds, negative_preds, constraints)
        loss.backward()
        # this gradient clipping leads to lower results, so I removed it
        # torch.nn.utils.clip_grad_value_(self.ncr_net.parameters(), 50)  # this has been inserted in the code provided
        self.optimizer.step()
        return loss.item()


    def get_state(self):
        state = {
            'epoch': self.current_epoch,
            'network': self.ncr_net.get_state(),
            'optimizer': self.optimizer.state_dict(),
            'params': {
                 'logic_reg_weight' : self.reg_weight,
                 'opt_conf' : self.opt_conf
            }
        }
        return state


    @classmethod
    def from_state(cls, state):
        ncr_net_class = getattr(import_module(cls.__module__), state["network"]["name"])
        ncr_net = ncr_net_class(**state['network']['params'])
        trainer = NCR_trainer(ncr_net, **state['params'])
        trainer.ncr_net.load_state_dict(state["network"]['state'])
        trainer.optimizer.load_state_dict(state['optimizer'])
        trainer.current_epoch = state['epoch']
        return trainer


class NCR(NeuralModel):
    r"""CFGAN: A Generic Collaborative Filtering Framework based on Generative Adversarial Networks.

    Parameters
    ----------
    gan_layers_dim : :obj:`list` of :obj:`int`
        The dimension of the layers of the generator network ordered from the input to the output.
    dic_layers_dim : :obj:`list` of :obj:`int`
        The dimension of the layers of the discriminator network ordered from the input to the
        output.
    alpha : :obj:`float` [optional]
        The ZR-coefficient, by default 0.1.
    s_pm : :obj:`float` [optional]
        The sampling parameter for the partial-masking approach, by default 0.7.
    s_zr : :obj:`float` [optional]
        The sampling parameter for the zero-reconstruction regularization, by default 0.5.
    opt_conf : :obj:`dict` [optional]
        The optimizer configuration dictionary, by default :obj:`None`.
    device : :obj:`str` [optional]
        The device where the model must be loaded, by default :obj:`None`. If :obj:`None`, the
        default device (see `rectorch.env.device`) is used.
    trainer : :class:`rectorch.models.nn.multvae.CFGAN_trainer` [optional]
        The trainer object for performing the learning, by default :obj:`None`. If not :obj:`None`
        it is the only parameters that is taken into account for creating the model.
    """
    def __init__(self,
                 n_users,
                 n_items,
                 emb_size=64,
                 dropout=0.0,
                 remove_double_not=False,
                 logic_reg_weight=0.01,
                 opt_conf=None,
                 device=None,
                 trainer=None):
        if trainer is not None:
            super(NCR, self).__init__(trainer.net, trainer, trainer.device)
        else:
            device = torch.device(device) if device is not None else env.device
            network = NCR_net(n_users, n_items, emb_size, dropout, remove_double_not)
            trainer = NCR_trainer(network,
                                  logic_reg_weight,
                                  device=device,
                                  opt_conf=opt_conf)
            super(NCR, self).__init__(network, trainer, device)


    def train(self,
              dataset,
              batch_size=128,
              n_neg_samples_t=1,
              n_neg_samples_vt=100,
              shuffle=True,
              valid_metric=None,
              valid_func=ValidFunc(logic_evaluate),
              num_epochs=100,
              at_least=20,
              early_stop=5,
              best_path="./saved_models/best_ncr_model.json",
              verbose=1,
              seed=None):
        r"""Training procedure of CFGAN.

        The training works in an alternate way between generator and discriminator.
        The number of training batches in each epochs are ``g_steps`` and ``d_steps``, respectively.

        Parameters
        ----------
        dataset : :class:`rectorch.data.Dataset` or :class:`rectorch.samplers.CFGAN_TrainingSampler`
            The dataset/sampler object that load the training/validation set in mini-batches.
        batch_size : :obj:`int` [optional]
            The size of the batches, by default 64.
        valid_metric : :obj:`str` [optional]
            The metric used during the validation to select the best model, by default :obj:`None`.
            If ``valid_data`` is not :obj:`None` then ``valid_metric`` must be not :obj:`None`.
            To see the valid strings for the metric please see the module :mod:`metrics`.
        valid_func : :class:`rectorch.validation.ValidFunc` [optional]
            The validation function, by default a standard validation procedure, i.e.,
            :func:`rectorch.evaluation.evaluate`.
        num_epochs : :obj:`int` [optional]
            Number of training epochs, by default 1000.
        g_steps : :obj:`int` [optional]
            Number of steps for a generator epoch, by default 5.
        d_steps : :obj:`int` [optional]
            Number of steps for a discriminator epoch, by default 5.
        verbose : :obj:`int` [optional]
            The level of verbosity of the logging, by default 1. The level can have any integer
            value greater than 0. However, after reaching a maximum verbosity value (that depends on
            the size of the training set), higher values will not have any effect.
        """
        set_seed(seed)
        if isinstance(dataset, Sampler):
            data_sampler = dataset
        else:
            data_sampler = NCR_Sampler(dataset,
                                       mode="train",
                                       n_neg_samples_t=n_neg_samples_t,
                                       n_neg_samples_vt=n_neg_samples_vt,
                                       batch_size=batch_size,
                                       shuffle=shuffle)

        best_val = 0.0
        early_stop_counter = 0
        early_stop_flag = False
        if early_stop > 1:  # it means that the parameter is meaningful
            early_stop_flag = True
        try:
            for epoch in range(1, num_epochs + 1):
                self.trainer.train_epoch(epoch, data_sampler, verbose)
                if valid_metric is not None:
                    data_sampler.valid()
                    valid_res = valid_func(self, data_sampler, valid_metric)
                    mu_val = np.mean(valid_res)
                    std_err_val = np.std(valid_res) / np.sqrt(len(valid_res))
                    env.logger.info('| epoch %d | %s %.3f (%.4f) |',
                                epoch, valid_metric, mu_val, std_err_val)
                    if mu_val > best_val:
                        best_val = mu_val
                        self.save_model(best_path)  # save model if an improved validation score has been
                        # obtained
                        early_stop_counter = 0  # we have reached a new validation best value, so we put the early stop
                        # counter to zero
                    else:
                        # if we did not have obtained an improved validation metric, we have to increase the early
                        # stopping counter
                        if epoch >= at_least and early_stop_flag:  # we have to train for at least 20 epochs, they said that in the paper
                            early_stop_counter += 1
                            if early_stop_counter == early_stop:
                                env.logger.info('Traing stopped at epoch %d due to early stopping', epoch)
                                break
        except KeyboardInterrupt:
            env.logger.warning('Handled KeyboardInterrupt: exiting from training early')

    def predict(self, batch_data):
        """Performs the prediction on the given batch using the trained NCR network. It takes as input a batch of
        logical expressions and it returns the predictions for the positive and negative logical expressions in the
        batch. Note that during validation we have one positive expression and 100 negative expressions for each
        interaction in the batch.
        Parameters
        ----------
        batch_data : :class:`torch.Tensor`
            The input for which the prediction has to be computed.
        Returns
        -------
        :return positive_predictions: the predictions for the positive logical expressions contained in the input batch.
        There is one positive prediction for each row of the batch.
        :return negative_predictions: the predictions for the negative logical expressions contained in the inpu batch.
        There are 100 negative predictions for each row of the batch.
        """
        self.network.eval()  # we have to set the network in evaluation mode
        with torch.no_grad():
            positive_predictions, negative_predictions, _ = self.network(batch_data)
        return positive_predictions, negative_predictions


    def test(self, dataset, n_neg_samples=100, batch_size=256, test_metrics=['ndcg@5', 'ndcg@10', 'hit@5', 'hit@10'],
             n_times=10):
        """
        This method performs the test of a trained NCR model.
        :param test_loader: this is the DataSampler that loads the test set interactions.
        :param test_metrics: this is a list containing the test metrics that have to be computed.
        :param n_times: this is the number of times that the evaluation has to be computed. Since the test loader
        generates 100 random negative items for each interaction in the test set, different random generations
        could lead to different test performances. The evaluation will be computed n_times times and then each metric
        will be averaged among these n_times evaluations.
        This method will log the value of each one of the metrics (plus std error) once this procedure has finished.
        """
        if isinstance(dataset, Sampler):
            data_sampler = dataset
        else:
            data_sampler = NCR_Sampler(dataset,
                                       mode="test",
                                       n_neg_samples_vt=n_neg_samples,
                                       batch_size=batch_size)
        data_sampler.test()
        metric_dict = {}
        for i in range(n_times):  # compute test metrics n_times times and take the mean since negative samples are
            # randomly generated
            evaluation_dict = logic_evaluate(self, data_sampler, test_metrics)
            for metric in evaluation_dict:
                if metric not in metric_dict:
                    metric_dict[metric] = {}
                metric_mean = np.mean(evaluation_dict[metric])
                metric_std_err_val = np.std(evaluation_dict[metric]) / np.sqrt(len(evaluation_dict[metric]))
                if "mean" not in metric_dict[metric]:
                    metric_dict[metric]["mean"] = metric_mean
                    metric_dict[metric]["std"] = metric_std_err_val
                else:
                    metric_dict[metric]["mean"] += metric_mean
                    metric_dict[metric]["std"] += metric_std_err_val

        for metric in metric_dict:
            env.logger.info('%s: %.3f (%.4f)', metric, metric_dict[metric]["mean"] / n_times,
                        metric_dict[metric]["std"] / n_times)

    @classmethod
    def from_state(cls, state):
        trainer = NCR_trainer.from_state(state)
        return NCR(trainer=trainer)

    @classmethod
    def load_model(cls, filepath, device=None):
        state = torch.load(filepath)
        if device:
            state["device"] = device
        return cls.from_state(state)
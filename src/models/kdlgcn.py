# -*- coding: utf-8 -*-
r"""
LightGCN
################################################

Reference:
    Xiangnan He et al. "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation." in SIGIR 2020.

Reference code:
    https://github.com/kuandeng/LightGCN
"""

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss, MSELoss
from common.init import xavier_uniform_initialization, xavier_normal_initialization


class KDLGCN(GeneralRecommender):
    r"""KDLGCN is a multimodal RS based on LightGCN and Knowledge Distillation from multimodal features.

    """
    def __init__(self, config, dataset):
        super(KDLGCN, self).__init__(config, dataset)

        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(
            form='coo').astype(np.float64)

        # load parameters info
        self.latent_dim = config['embedding_size']  # int type:the embedding size of lightGCN
        self.n_layers = config['n_layers']  # int type:the layer num of lightGCN
        self.reg_weight = config['reg_weight']  # float32 type: the weight decay for l2 normalizaton
        self.kd_weight = config['kd_weight']

        # generate intermediate data
        self.norm_adj_matrix = self.get_norm_adj_mat().to(self.device)

        # both modalities must be available
        assert self.v_feat is not None and self.t_feat is not None, 'error: both modalities are missing'

        # original multimodal features
        self.item_visual = self.v_feat
        self.item_textual = self.t_feat

        # linear multimodal features
        self.item_visual_linear = nn.Linear(self.item_visual.shape[1], self.latent_dim)
        self.item_textual_linear = nn.Linear(self.item_textual.shape[1], self.latent_dim)

        # KD, CF and regularization losses
        self.kd_loss = MSELoss()
        self.cf_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        # parameters initialization
        self.apply(xavier_uniform_initialization)
        self.embedding_dict = self._init_model()

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.n_users, self.latent_dim))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.n_items, self.latent_dim)))
        })

        return embedding_dict

    def get_norm_adj_mat(self):
        r"""Get the normalized interaction matrix of users and items.

        Construct the square matrix from the training data and normalize it
        using the laplace matrix.

        .. math::
            A_{hat} = D^{-0.5} \times A \times D^{-0.5}

        Returns:
            Sparse tensor of the normalized interaction matrix.
        """
        # build adj matrix
        A = sp.dok_matrix((self.n_users + self.n_items,
                           self.n_users + self.n_items), dtype=np.float64)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col+self.n_users),
                             [1]*inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row+self.n_users, inter_M_t.col),
                                  [1]*inter_M_t.nnz)))
        # ATTENTION: _update is no longer supported by scipy
        # A._update(data_dict)
        # here is the fixed version of the code
        for (row, col), value in data_dict.items():
            A[row, col] = value
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid Devide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor([row, col])
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
        return SparseL

    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.

        Returns:
            Tensor of the embedding matrix. Shape of [n_items+n_users, embedding_dim]
        """
        # user_embeddings = self.user_embedding.weight
        # item_embeddings = self.item_embedding.weight
        # ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        return ego_embeddings

    def forward(self):
        all_embeddings = self.get_ego_embeddings()
        embeddings_list = [all_embeddings]

        for layer_idx in range(self.n_layers):
            all_embeddings = torch.sparse.mm(self.norm_adj_matrix, all_embeddings)
            embeddings_list.append(all_embeddings)
        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

        user_all_embeddings = lightgcn_all_embeddings[:self.n_users, :]
        item_all_embeddings = lightgcn_all_embeddings[self.n_users:, :]

        return user_all_embeddings, item_all_embeddings

    def calculate_loss(self, interaction):
        user = interaction[0]
        pos_item = interaction[1]
        neg_item = interaction[2]

        user_all_embeddings, item_all_embeddings = self.forward()
        user_e = user_all_embeddings[user, :]
        pos_e = item_all_embeddings[pos_item, :]
        neg_e = item_all_embeddings[neg_item, :]

        # get visual and textual embeddings
        pos_visual_item = self.item_visual_linear(self.item_visual[pos_item, :])
        pos_textual_item = self.item_textual_linear(self.item_textual[pos_item, :])

        # compute KD losses between ID and modality embeddings

        # compute the three losses: MF loss (BPR), reg loss, KD loss (MSE)

        # reg loss based on ego embeddings
        u_ego_embeddings = self.embedding_dict['user_emb'][user, :]
        posi_ego_embeddings = self.embedding_dict['item_emb'][pos_item, :]
        negi_ego_embeddings = self.embedding_dict['item_emb'][neg_item, :]
        reg_loss = self.reg_weight * self.reg_loss(u_ego_embeddings, posi_ego_embeddings, negi_ego_embeddings)

        # BPR loss 
        pos_item_score, neg_item_score = torch.mul(user_e, pos_e).sum(dim=1), torch.mul(user_e, neg_e).sum(dim=1)
        mf_loss = self.cf_loss(pos_item_score, neg_item_score)
        
        # KD loss
        kd_visual = self.kd_loss(pos_visual_item, pos_e)
        kd_text = self.kd_loss(pos_textual_item, pos_e)
        kd_loss = self.kd_weight * (kd_text + kd_visual)

        # final loss
        loss = mf_loss + reg_loss + kd_loss
        return loss

        # # calculate BPR Loss
        # pos_scores = torch.mul(u_embeddings, posi_embeddings).sum(dim=1)
        # neg_scores = torch.mul(u_embeddings, negi_embeddings).sum(dim=1)
        # mf_loss = self.mf_loss(pos_scores, neg_scores)

        # # calculate BPR Loss
        # u_ego_embeddings = self.embedding_dict['user_emb'][user, :]
        # posi_ego_embeddings = self.embedding_dict['item_emb'][pos_item, :]
        # negi_ego_embeddings = self.embedding_dict['item_emb'][neg_item, :]

        # reg_loss = self.reg_loss(u_ego_embeddings, posi_ego_embeddings, negi_ego_embeddings)
        # loss = mf_loss + self.reg_weight * reg_loss

        # return loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        restore_user_e, restore_item_e = self.forward()
        u_embeddings = restore_user_e[user, :]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, restore_item_e.transpose(0, 1))

        return scores

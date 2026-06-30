# coding: utf-8
# @email: enoche.chow@gmail.com
r"""
VBPR -- Recommended version
################################################
Reference:
VBPR: Visual Bayesian Personalized Ranking from Implicit Feedback -Ruining He, Julian McAuley. AAAI'16
"""
import numpy as np
import os
import torch
import torch.nn as nn

from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss, MSELoss
from common.init import xavier_normal_initialization
import torch.nn.functional as F


class KDVBPR(GeneralRecommender):
    r"""BPR is a basic matrix factorization model that be trained in the pairwise way.
    """
    def __init__(self, config, dataloader):
        super(KDVBPR, self).__init__(config, dataloader)

        # load parameters info
        self.u_embedding_size = self.i_embedding_size = config['embedding_size']
        self.reg_weight = config['reg_weight']  # float32 type: the weight decay for l2 normalizaton
        self.kd_weight = config['kd_weight']

        # define layers and loss
        self.u_embedding = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.n_users, self.u_embedding_size)))
        self.i_embedding = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.n_items, self.i_embedding_size)))

        # both modalities must be available
        assert self.v_feat is not None and self.t_feat is not None, 'error: both modalities are missing'

        # original multimodal features
        self.item_visual = self.v_feat
        self.item_textual = self.t_feat

        # linear multimodal features
        self.item_visual_linear = nn.Linear(self.item_visual.shape[1], self.i_embedding_size)
        self.item_textual_linear = nn.Linear(self.item_textual.shape[1], self.i_embedding_size)

        # KD, CF and regularization losses
        self.kd_loss = MSELoss()
        self.cf_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        # parameters initialization
        self.apply(xavier_normal_initialization)

    def get_user_embedding(self, user):
        r""" Get a batch of user embedding tensor according to input user's id.

        Args:
            user (torch.LongTensor): The input tensor that contains user's id, shape: [batch_size, ]

        Returns:
            torch.FloatTensor: The embedding tensor of a batch of user, shape: [batch_size, embedding_size]
        """
        return self.u_embedding[user, :]

    def get_item_embedding(self, item):
        r""" Get a batch of item embedding tensor according to input item's id.

        Args:
            item (torch.LongTensor): The input tensor that contains item's id, shape: [batch_size, ]

        Returns:
            torch.FloatTensor: The embedding tensor of a batch of item, shape: [batch_size, embedding_size]
        """
        return self.item_embedding[item, :]

    def forward(self, dropout=0.0):
        # item_embeddings = self.item_linear(self.item_raw_features)
        # item_embeddings = torch.cat((self.i_embedding, item_embeddings), -1)

        user_e = F.dropout(self.u_embedding, dropout)
        item_e = F.dropout(self.i_embedding, dropout)
        return user_e, item_e

    def calculate_loss(self, interaction):
        """
        loss on one batch
        :param interaction:
            batch data format: tensor(3, batch_size)
            [0]: user list; [1]: positive items; [2]: negative items
        :return:
        """
        user = interaction[0]
        pos_item = interaction[1]
        neg_item = interaction[2]

        # get ID embeddings
        user_embeddings, item_embeddings = self.forward()
        user_e = user_embeddings[user, :]
        pos_e = item_embeddings[pos_item, :]
        neg_e = item_embeddings[neg_item, :]

        # get visual and textual embeddings
        pos_visual_item = self.item_visual_linear(self.item_visual[pos_item, :])
        pos_textual_item = self.item_textual_linear(self.item_textual[pos_item, :])

        # compute KD losses between ID and modality embeddings
        kd_visual = self.kd_loss(pos_visual_item, pos_e)
        kd_text = self.kd_loss(pos_textual_item, pos_e)

        # compute losses
        pos_item_score, neg_item_score = torch.mul(user_e, pos_e).sum(dim=1), torch.mul(user_e, neg_e).sum(dim=1)
        mf_loss = self.cf_loss(pos_item_score, neg_item_score)
        reg_loss = self.reg_weight * self.reg_loss(user_e, pos_e, neg_e)
        kd_loss = self.kd_weight * (kd_text + kd_visual)

        loss = mf_loss + reg_loss + kd_loss
        return loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        user_embeddings, item_embeddings = self.forward()
        user_e = user_embeddings[user, :]
        all_item_e = item_embeddings
        score = torch.matmul(user_e, all_item_e.transpose(0, 1))
        return score

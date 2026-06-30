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
from common.loss import BPRLoss, EmbLoss
from common.init import xavier_normal_initialization
import torch.nn.functional as F

# import fusion classes
from common.fusion_layers import ConcatFusionLayer, LinearSumFusionLayer, GatingFusionLayer


class Fusion_VBPR(GeneralRecommender):
    r"""BPR is a basic matrix factorization model that be trained in the pairwise way.
    """
    def __init__(self, config, dataloader):
        super(Fusion_VBPR, self).__init__(config, dataloader)

        # load parameters info
        self.u_embedding_size = self.i_embedding_size = config['embedding_size']
        self.reg_weight = config['reg_weight']
        # self.reg_weight = 0.1
        
        # define layers and loss
        self.u_embedding = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.n_users, self.u_embedding_size * 2)))
        self.i_embedding = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.n_items, self.i_embedding_size)))

        # check that the fusion is possible (that is, both modalities are available)
        assert self.v_feat is not None, "Missing visual modality, no fusion is possible"
        assert self.t_feat is not None, "Textual  visual modality, no fusion is possible"

        # check that the fusion strategy is available
        # available_fusion_strategy = ['concat', 'linear_sum']
        # assert config['fusion_strategy'] in available_fusion_strategy, "The selected fusion strategy is not available"

        # fusion layer
        self.fusion_strategy = config['fusion_layer']
        if self.fusion_strategy == 'concat':
            self.fusion_layer = ConcatFusionLayer([self.v_feat, self.t_feat], self.i_embedding_size)
        elif self.fusion_strategy == 'linear_sum':
            self.fusion_layer = LinearSumFusionLayer([self.v_feat, self.t_feat], self.i_embedding_size)
        elif self.fusion_strategy == 'gating':
            self.fusion_layer = GatingFusionLayer([self.v_feat, self.t_feat], self.i_embedding_size)
        
        self.loss = BPRLoss()
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
        # item_embeddings = self.item_linear(self.item_raw_features)
        # item_embeddings = torch.cat((self.i_embedding, item_embeddings), -1)

        if self.fusion_strategy == 'gating':
            item_embeddings = self.fusion_layer.forward(self.i_embedding)
        else:
            item_embeddings = self.fusion_layer.forward()
        item_embeddings = torch.cat((self.i_embedding, item_embeddings), -1)

        user_e = F.dropout(self.u_embedding, dropout)
        item_e = F.dropout(item_embeddings, dropout)
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

        user_embeddings, item_embeddings = self.forward()
        user_e = user_embeddings[user, :]
        pos_e = item_embeddings[pos_item, :]
        #neg_e = self.get_item_embedding(neg_item)
        neg_e = item_embeddings[neg_item, :]
        pos_item_score, neg_item_score = torch.mul(user_e, pos_e).sum(dim=1), torch.mul(user_e, neg_e).sum(dim=1)
        mf_loss = self.loss(pos_item_score, neg_item_score)
        reg_loss = self.reg_loss(user_e, pos_e, neg_e)
        loss = mf_loss + self.reg_weight * reg_loss
        return loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        user_embeddings, item_embeddings = self.forward()
        user_e = user_embeddings[user, :]
        all_item_e = item_embeddings
        score = torch.matmul(user_e, all_item_e.transpose(0, 1))
        return score

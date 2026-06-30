# # coding: utf-8
# # @email: enoche.chow@gmail.com
# r"""
# VBPR -- Recommended version
# ################################################
# Reference:
# VBPR: Visual Bayesian Personalized Ranking from Implicit Feedback -Ruining He, Julian McAuley. AAAI'16
# """
# import numpy as np
# import os
# import torch
# import torch.nn as nn

# from common.abstract_recommender import GeneralRecommender
# from common.loss import BPRLoss, EmbLoss
# from common.init import xavier_normal_initialization
# import torch.nn.functional as F


# class MMBPR(GeneralRecommender):
#     r"""BPR is a basic matrix factorization model that be trained in the pairwise way.
#     """
#     def __init__(self, config, dataloader):
#         super(MMBPR, self).__init__(config, dataloader)

#         # load parameters info
#         self.u_embedding_size = self.i_embedding_size = config['embedding_size']
#         self.reg_weight = config['reg_weight']  # float32 type: the weight decay for l2 normalizaton
#         if self.reg_weight is None:
#             self.reg_weight = 1e-4

#         # load multimodal features
#         if self.v_feat is not None and self.t_feat is not None:
#             print('Using visual and text features')
#             self.item_raw_features = torch.cat((self.t_feat, self.v_feat), -1)
#         elif self.v_feat is not None:
#             print('Using visual features ONLY')
#             self.item_raw_features = self.v_feat
#         else:
#             print('Using textual features ONLY')
#             self.item_raw_features = self.t_feat
#         print(self.item_raw_features.shape)

#         self.user_preferences = dataloader._get_history_items_u()
        

#         # user is obtained as the sum of the items the user liked, projected to the emb size
#         # item is just the item embs, projected to the emb size
#         self.user_linear = nn.Linear(self.item_raw_features.shape[1], self.u_embedding_size)
#         self.item_linear = nn.Linear(self.item_raw_features.shape[1], self.i_embedding_size)
#         self.loss = BPRLoss()
#         self.reg_loss = EmbLoss()

#         # parameters initialization
#         self.apply(xavier_normal_initialization)

#     def get_user_embedding(self, user):
#         r""" Get a batch of user embedding tensor according to input user's id.

#         Args:
#             user (torch.LongTensor): The input tensor that contains user's id, shape: [batch_size, ]

#         Returns:
#             torch.FloatTensor: The embedding tensor of a batch of user, shape: [batch_size, embedding_size]
#         """

#         batch_user_embs = []

#         for u in user:
#             user_items = torch.tensor(list(self.user_preferences[u.item()]), device=self.device)
#             item_feats = self.item_raw_features[user_items]  # Shape: (num_items_for_u, feature_dim)
#             user_emb = item_feats.sum(dim=0)  # Sum over items -> (feature_dim,)
#             batch_user_embs.append(user_emb)

#         # Stack into a batch tensor: (batch_size, feature_dim)
#         batch_user_embs = torch.stack(batch_user_embs, dim=0)

#         return self.user_linear(batch_user_embs)

#     def get_item_embedding(self, item):
#         r""" Get a batch of item embedding tensor according to input item's id.

#         Args:
#             item (torch.LongTensor): The input tensor that contains item's id, shape: [batch_size, ]

#         Returns:
#             torch.FloatTensor: The embedding tensor of a batch of item, shape: [batch_size, embedding_size]
#         """
#         return self.item_linear(self.item_raw_features[item])

#     def forward(self):

#         # avg of the interactions
#         batch_user_embs = []
#         for u in self.user_preferences:
#             user_items = torch.tensor(list(self.user_preferences[u]), device=self.device)
#             item_feats = self.item_raw_features[user_items]
#             user_emb = item_feats.sum(dim=0)
#             batch_user_embs.append(user_emb)

#         # get the resulting embeddings
#         user_embs = torch.stack(batch_user_embs, dim=0)
#         item_embs = self.item_raw_features

#         return self.user_linear(user_embs), self.item_linear(item_embs)

#     def calculate_loss(self, interaction):
#         """
#         loss on one batch
#         :param interaction:
#             batch data format: tensor(3, batch_size)
#             [0]: user list; [1]: positive items; [2]: negative items
#         :return:
#         """
#         user = interaction[0]
#         pos_item = interaction[1]
#         neg_item = interaction[2]

#         user_embs, item_embs = self.forward()

#         user_e = user_embs[user, :]
#         pos_e = item_embs[pos_item, :]
#         neg_e = item_embs[neg_item, :]
        
#         pos_item_score, neg_item_score = torch.mul(user_e, pos_e).sum(dim=1), torch.mul(user_e, neg_e).sum(dim=1)
#         mf_loss = self.loss(pos_item_score, neg_item_score)
#         reg_loss = self.reg_loss(user_e, pos_e, neg_e)
#         loss = mf_loss + self.reg_weight * reg_loss
#         return loss

#     def full_sort_predict(self, interaction):
#         user = interaction[0]
#         user_embeddings, item_embeddings = self.forward()
#         user_e = user_embeddings[user, :]
#         all_item_e = item_embeddings
#         score = torch.matmul(user_e, all_item_e.transpose(0, 1))
#         return score



import numpy as np
import torch
import torch.nn as nn
from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss
from common.init import xavier_normal_initialization


class MMBPR(GeneralRecommender):
    def __init__(self, config, dataloader):
        super(MMBPR, self).__init__(config, dataloader)

        self.embedding_size = config['embedding_size']
        self.reg_weight = config['reg_weight']  # float32 type: the weight decay for l2 normalizaton
        if self.reg_weight is None:
            self.reg_weight = 1e-4
        self.aggregation = config['aggregation']

        # Select multimodal features
        if self.v_feat is not None and self.t_feat is not None:
            self.item_raw_features = torch.cat((self.t_feat, self.v_feat), dim=-1)
        elif self.v_feat is not None:
            self.item_raw_features = self.v_feat
        else:
            self.item_raw_features = self.t_feat
        print(f"Item raw feature shape: {self.item_raw_features.shape}")

        # Register item features as buffer (no gradient)
        self.register_buffer("item_raw_features_buffer", self.item_raw_features)

        # Linear projection layers
        # self.user_linear = nn.Linear(self.item_raw_features.shape[1], self.embedding_size)
        # self.item_linear = nn.Linear(self.item_raw_features.shape[1], self.embedding_size)
        self.user_linear = nn.Linear(self.item_raw_features.shape[1], self.item_raw_features.shape[1])
        self.item_linear = nn.Linear(self.item_raw_features.shape[1], self.item_raw_features.shape[1])

        self.loss = BPRLoss()
        self.reg_loss = EmbLoss()

        self.apply(xavier_normal_initialization)

        # Pre-cache user item history as tensors
        self.user_histories = {
            u: torch.tensor(list(items), dtype=torch.long, device=self.device)
            for u, items in dataloader._get_history_items_u().items()
        }

    def get_user_embedding(self, users):
        """
        Efficient batched user embedding computation
        """
        batch_size = len(users)
        max_items = max(len(self.user_histories[u.item()]) for u in users)

        padded_items = torch.zeros((batch_size, max_items), dtype=torch.long, device=self.device)
        mask = torch.zeros((batch_size, max_items), dtype=torch.float32, device=self.device)

        for i, u in enumerate(users):
            items = self.user_histories[u.item()]
            length = len(items)
            padded_items[i, :length] = items
            mask[i, :length] = 1.0

        item_feats = self.item_raw_features_buffer[padded_items]  # (B, max_len, feat_dim)
        masked_feats = item_feats * mask.unsqueeze(-1)

        if self.aggregation == 'sum':
            user_feat_agg = masked_feats.sum(dim=1)
        elif self.aggregation == 'avg':
            user_feat_agg = masked_feats.mean(dim=1)
        else:
            raise Exception(f'Aggregation not available!')
        return self.user_linear(user_feat_agg)

    def get_item_embedding(self, items):
        return self.item_linear(self.item_raw_features_buffer[items])

    def forward(self):
        """
        Full forward: returns all user and item embeddings
        """
        all_user_ids = list(self.user_histories.keys())
        user_tensor = torch.tensor(all_user_ids, dtype=torch.long, device=self.device)
        user_embs = self.get_user_embedding(user_tensor)
        item_embs = self.item_linear(self.item_raw_features_buffer)
        return user_embs, item_embs

    def calculate_loss(self, interaction):
        user = interaction[0]
        pos_item = interaction[1]
        neg_item = interaction[2]

        user_emb = self.get_user_embedding(user)
        pos_emb = self.get_item_embedding(pos_item)
        neg_emb = self.get_item_embedding(neg_item)

        pos_scores = torch.sum(user_emb * pos_emb, dim=1)
        neg_scores = torch.sum(user_emb * neg_emb, dim=1)

        mf_loss = self.loss(pos_scores, neg_scores)
        reg_loss = self.reg_loss(user_emb, pos_emb, neg_emb)
        return mf_loss + self.reg_weight * reg_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        user_emb = self.get_user_embedding(user)
        item_emb = self.item_linear(self.item_raw_features_buffer)
        return torch.matmul(user_emb, item_emb.T)

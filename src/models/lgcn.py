# -*- coding: utf-8 -*-
r"""
LightGCN (PyG Implementation Style)
################################################

Reference:
    Xiangnan He et al. "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation." in SIGIR 2020.

Reference code:
    https://github.com/pyg-team/pytorch_geometric
"""

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss


class LGConv(nn.Module):
    r"""Light Graph Convolution layer from LightGCN.
    
    Performs the propagation: x^(l+1) = D^(-0.5) * A * D^(-0.5) * x^(l)
    """
    def __init__(self):
        super(LGConv, self).__init__()
    
    def forward(self, x, edge_index):
        """
        Args:
            x: Node features [N, D]
            edge_index: Normalized sparse adjacency matrix
        
        Returns:
            Propagated node features [N, D]
        """
        return torch.sparse.mm(edge_index, x)


class LGCN(GeneralRecommender):
    r"""LightGCN with PyG-style implementation.

    This implementation follows PyTorch Geometric's approach:
    - Configurable alpha weights for layer aggregation
    - Modular LGConv layers
    - Flexible embedding aggregation strategy
    
    The final embedding is computed as:
        x_i = sum_{l=0}^{L} alpha_l * x_i^(l)
    
    where alpha weights can be configured or default to uniform: 1/(L+1)
    """
    def __init__(self, config, dataset):
        super(LGCN, self).__init__(config, dataset)

        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(
            form='coo').astype(np.float32)

        # load parameters info
        self.embedding_dim = config['embedding_size']
        self.n_layers = config['n_layers']
        self.reg_weight = config['reg_weight']
        
        # alpha weights for layer aggregation (PyG style)
        # If None, use uniform: 1/(n_layers+1) for each layer
        alpha = config['alpha']
        if alpha == -1:
            alpha = 1.0 / (self.n_layers + 1)
        
        if isinstance(alpha, (list, tuple)):
            assert len(alpha) == self.n_layers + 1, \
                f"alpha must have length {self.n_layers + 1}"
            alpha = torch.tensor(alpha, dtype=torch.float32)
        else:
            alpha = torch.tensor([alpha] * (self.n_layers + 1), dtype=torch.float32)
        
        # self.register_buffer('alpha', alpha)
        self.alpha = alpha

        # Initialize embeddings (single embedding table for all nodes)
        self.num_nodes = self.n_users + self.n_items
        self.embedding = nn.Embedding(self.num_nodes, self.embedding_dim)
        
        # LGConv layers (PyG style)
        self.convs = nn.ModuleList([LGConv() for _ in range(self.n_layers)])

        # Loss functions
        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        # Generate normalized adjacency matrix
        self.norm_adj_matrix = self.get_norm_adj_mat().to(self.device)

        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        """Reset all learnable parameters (PyG style)."""
        nn.init.xavier_uniform_(self.embedding.weight)

    def get_norm_adj_mat(self):
        r"""Get the normalized interaction matrix of users and items.

        Construct the square matrix from the training data and normalize it
        using the laplace matrix.

        .. math::
            A_{hat} = D^{-0.5} \times A \times D^{-0.5}

        Returns:
            Sparse tensor of the normalized interaction matrix.
        """
        # Build adjacency matrix
        A = sp.dok_matrix((self.n_users + self.n_items,
                           self.n_users + self.n_items), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users),
                             [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col),
                                  [1] * inter_M_t.nnz)))
        # A._update(data_dict)
        # here is the fixed version of the code
        for (row, col), value in data_dict.items():
            A[row, col] = value
        
        # Normalize adjacency matrix
        sumArr = (A > 0).sum(axis=1)
        # Add epsilon to avoid divide by zero
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        
        # Convert to sparse tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor([row, col])
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
        return SparseL

    def get_embedding(self):
        r"""Get embeddings using PyG-style weighted layer aggregation.
        
        Returns the weighted sum of embeddings from all layers:
            out = sum_{l=0}^{L} alpha_l * x^(l)
        
        Returns:
            Tensor of shape [n_users + n_items, embedding_dim]
        """
        x = self.embedding.weight
        out = x * self.alpha[0]  # Layer 0 (initial embeddings)

        for i in range(self.n_layers):
            x = self.convs[i](x, self.norm_adj_matrix)
            out = out + x * self.alpha[i + 1]

        return out

    def forward(self):
        r"""Forward pass to get user and item embeddings.
        
        Returns:
            Tuple of (user_embeddings, item_embeddings)
        """
        all_embeddings = self.get_embedding()
        
        user_all_embeddings = all_embeddings[:self.n_users, :]
        item_all_embeddings = all_embeddings[self.n_users:, :]

        return user_all_embeddings, item_all_embeddings

    def calculate_loss(self, interaction):
        r"""Calculate BPR loss with L2 regularization.
        
        Args:
            interaction: Tuple of (user, pos_item, neg_item) indices
            
        Returns:
            Total loss (BPR loss + regularization)
        """
        user = interaction[0]
        pos_item = interaction[1]
        neg_item = interaction[2]

        user_all_embeddings, item_all_embeddings = self.forward()

        u_embeddings = user_all_embeddings[user, :]
        posi_embeddings = item_all_embeddings[pos_item, :]
        negi_embeddings = item_all_embeddings[neg_item, :]

        # Calculate BPR Loss
        pos_scores = torch.mul(u_embeddings, posi_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, negi_embeddings).sum(dim=1)
        mf_loss = self.mf_loss(pos_scores, neg_scores)

        # Calculate regularization on initial embeddings (ego embeddings)
        # Note: In PyG style, we regularize the initial embeddings
        u_ego_embeddings = self.embedding.weight[user, :]
        posi_ego_embeddings = self.embedding.weight[self.n_users + pos_item, :]
        negi_ego_embeddings = self.embedding.weight[self.n_users + neg_item, :]

        reg_loss = self.reg_loss(u_ego_embeddings, posi_ego_embeddings, negi_ego_embeddings)
        
        loss = mf_loss + self.reg_weight * reg_loss

        return loss

    def full_sort_predict(self, interaction):
        r"""Predict scores for all items for given users.
        
        Args:
            interaction: User indices
            
        Returns:
            Score matrix [batch_size, n_items]
        """
        user = interaction[0]
        restore_user_e, restore_item_e = self.forward()
        u_embeddings = restore_user_e[user, :]

        # Dot product with all item embeddings
        scores = torch.matmul(u_embeddings, restore_item_e.transpose(0, 1))

        return scores
    
    def recommend(self, interaction, k=10, sorted=True):
        r"""Get top-k recommendations for users (PyG style API).
        
        Args:
            interaction: User indices
            k: Number of recommendations
            sorted: Whether to sort by score
            
        Returns:
            Top-k item indices for each user [batch_size, k]
        """
        scores = self.full_sort_predict(interaction)
        top_k_indices = scores.topk(k, dim=-1, sorted=sorted).indices
        return top_k_indices

    def predict_link(self, user_indices, item_indices, prob=False):
        r"""Predict links between specific user-item pairs (PyG style API).
        
        Args:
            user_indices: User indices
            item_indices: Item indices  
            prob: Whether to return probabilities (with sigmoid)
            
        Returns:
            Predictions for each user-item pair
        """
        user_all_embeddings, item_all_embeddings = self.forward()
        
        u_embeddings = user_all_embeddings[user_indices, :]
        i_embeddings = item_all_embeddings[item_indices, :]
        
        pred = (u_embeddings * i_embeddings).sum(dim=-1)
        
        if prob:
            pred = torch.sigmoid(pred)
        
        return pred

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.num_nodes}, '
                f'{self.embedding_dim}, num_layers={self.n_layers})')
# coding: utf-8
# Improved KDVBPR with better multimodal integration

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss, MSELoss
from common.init import xavier_normal_initialization


class KDMMCF(GeneralRecommender):
    """
    Enhanced KD_MMCF with:
    - Multimodal fusion for prediction
    - Attention-based feature integration
    - Improved KD loss strategy
    """
    def __init__(self, config, dataloader):
        super(KDMMCF, self).__init__(config, dataloader)

        # load parameters
        self.u_embedding_size = self.i_embedding_size = config['embedding_size']
        self.reg_weight = config['reg_weight']
        self.kd_weight = config['kd_weight']
        self.fusion_mode = config['fusion_mode']
        self.use_modality_in_pred = config['use_modality_in_pred']
        self.modality_dropout = config['modality_dropout']
        self.use_contrastive = config['use_contrastive']
        
        # embeddings
        self.u_embedding = nn.Parameter(
            nn.init.xavier_uniform_(torch.empty(self.n_users, self.u_embedding_size))
        )
        self.i_embedding = nn.Parameter(
            nn.init.xavier_uniform_(torch.empty(self.n_items, self.i_embedding_size))
        )

        # multimodal features
        assert self.v_feat is not None and self.t_feat is not None, \
            'error: both modalities are missing'
        
        self.item_visual = self.v_feat
        self.item_textual = self.t_feat

        # improved projection layers with batch normalization
        self.visual_projection = nn.Sequential(
            nn.Linear(self.item_visual.shape[1], self.i_embedding_size),
            nn.BatchNorm1d(self.i_embedding_size),
            nn.ReLU(),
            nn.Dropout(self.modality_dropout)
        )
        
        self.textual_projection = nn.Sequential(
            nn.Linear(self.item_textual.shape[1], self.i_embedding_size),
            nn.BatchNorm1d(self.i_embedding_size),
            nn.ReLU(),
            nn.Dropout(self.modality_dropout)
        )

        # fusion mechanisms
        if self.fusion_mode == 'attention':
            self.attention_layer = nn.Sequential(
                nn.Linear(self.i_embedding_size * 3, self.i_embedding_size),
                nn.Tanh(),
                nn.Linear(self.i_embedding_size, 3),
                nn.Softmax(dim=-1)
            )
        elif self.fusion_mode == 'gate':
            self.gate_visual = nn.Linear(self.i_embedding_size * 2, self.i_embedding_size)
            self.gate_textual = nn.Linear(self.i_embedding_size * 2, self.i_embedding_size)
        
        # loss functions
        self.kd_loss = MSELoss()
        self.cf_loss = BPRLoss()
        self.reg_loss = EmbLoss()
        
        # optional: contrastive loss for cross-modal alignment
        if self.use_contrastive:
            self.temperature = config['cl_temperature']

        self.apply(xavier_normal_initialization)
        
        # precompute and cache all item multimodal embeddings
        self._precompute_item_features()

    def _precompute_item_features(self):
        """Precompute visual and textual embeddings for all items"""
        with torch.no_grad():
            # Ensure features are on the same device as model
            device = self.i_embedding.device
            visual_feat = self.item_visual.to(device)
            textual_feat = self.item_textual.to(device)
            
            self.all_visual_e = self.visual_projection(visual_feat)
            self.all_textual_e = self.textual_projection(textual_feat)

    def get_multimodal_item_embedding(self, item):
        """Get fused item embeddings with multimodal features"""
        id_e = self.i_embedding[item, :].to(self.device)
        visual_e = self.all_visual_e[item, :].to(self.device)
        textual_e = self.all_textual_e[item, :].to(self.device)
        
        if not self.use_modality_in_pred:
            return id_e
        
        return self._fuse_embeddings(id_e, visual_e, textual_e)
    
    def _fuse_embeddings(self, id_e, visual_e, textual_e):
        """Fuse ID, visual, and textual embeddings"""
        # Ensure all tensors are on the same device
        device = id_e.device
        visual_e = visual_e.to(device)
        textual_e = textual_e.to(device)
        
        if self.fusion_mode == 'concat':
            # Simple concatenation + projection
            fused = torch.cat([id_e, visual_e, textual_e], dim=-1)
            # Need a projection layer for this - add if using
            return id_e + 0.3 * visual_e + 0.3 * textual_e
        
        elif self.fusion_mode == 'attention':
            # Attention-based fusion
            combined = torch.cat([id_e, visual_e, textual_e], dim=-1)
            attention_weights = self.attention_layer(combined)  # [batch, 3]
            
            stacked = torch.stack([id_e, visual_e, textual_e], dim=-1)  # [batch, dim, 3]
            fused = torch.sum(stacked * attention_weights.unsqueeze(1), dim=-1)
            return fused
        
        elif self.fusion_mode == 'gate':
            # Gated fusion
            gate_v = torch.sigmoid(self.gate_visual(torch.cat([id_e, visual_e], dim=-1)))
            gate_t = torch.sigmoid(self.gate_textual(torch.cat([id_e, textual_e], dim=-1)))
            return id_e + gate_v * visual_e + gate_t * textual_e
        
        else:  # default: weighted sum
            return id_e + 0.3 * visual_e + 0.3 * textual_e

    def forward(self, dropout=0.0):
        """Forward pass returning user and enhanced item embeddings"""
        user_e = F.dropout(self.u_embedding, dropout)
        item_e = F.dropout(self.i_embedding, dropout)
        return user_e, item_e

    def contrastive_loss(self, anchor, positive, negative=None):
        """Compute contrastive loss for cross-modal alignment"""
        # Normalize embeddings
        anchor = F.normalize(anchor, dim=-1)
        positive = F.normalize(positive, dim=-1)
        
        # Positive similarity
        pos_sim = torch.sum(anchor * positive, dim=-1) / self.temperature
        
        if negative is not None:
            negative = F.normalize(negative, dim=-1)
            neg_sim = torch.sum(anchor * negative, dim=-1) / self.temperature
            loss = -torch.log(torch.exp(pos_sim) / (torch.exp(pos_sim) + torch.exp(neg_sim)))
        else:
            # InfoNCE-style loss
            loss = -pos_sim
        
        return loss.mean()

    def calculate_loss(self, interaction):
        """Compute training loss with improved KD strategy"""
        user = interaction[0]
        pos_item = interaction[1]
        neg_item = interaction[2]

        # Get embeddings
        user_embeddings, item_embeddings = self.forward()
        user_e = user_embeddings[user, :]
        pos_id_e = item_embeddings[pos_item, :]
        neg_id_e = item_embeddings[neg_item, :]

        # Get multimodal embeddings (computed on-the-fly during training)
        device = user_e.device
        pos_visual_e = self.visual_projection(self.item_visual[pos_item, :].to(device))
        pos_textual_e = self.textual_projection(self.item_textual[pos_item, :].to(device))
        neg_visual_e = self.visual_projection(self.item_visual[neg_item, :].to(device))
        neg_textual_e = self.textual_projection(self.item_textual[neg_item, :].to(device))


        # Fuse embeddings for scoring
        pos_e = self._fuse_embeddings(pos_id_e, pos_visual_e, pos_textual_e)
        neg_e = self._fuse_embeddings(neg_id_e, neg_visual_e, neg_textual_e)

        # BPR loss
        pos_score = torch.mul(user_e, pos_e).sum(dim=1)
        neg_score = torch.mul(user_e, neg_e).sum(dim=1)
        mf_loss = self.cf_loss(pos_score, neg_score)

        # Improved KD loss: align modalities with ID embeddings
        # For positive items
        kd_visual_pos = self.kd_loss(pos_visual_e, pos_id_e.detach())
        kd_text_pos = self.kd_loss(pos_textual_e, pos_id_e.detach())
        
        # For negative items (optional but helpful)
        kd_visual_neg = self.kd_loss(neg_visual_e, neg_id_e.detach())
        kd_text_neg = self.kd_loss(neg_textual_e, neg_id_e.detach())
        
        kd_loss = self.kd_weight * (kd_visual_pos + kd_text_pos + 
                                    0.5 * (kd_visual_neg + kd_text_neg))

        # Contrastive loss for cross-modal alignment (optional)
        contrastive_loss = 0
        if self.use_contrastive:
            contrastive_loss = self.contrastive_loss(pos_visual_e, pos_textual_e) + \
                              self.contrastive_loss(pos_id_e, pos_visual_e) + \
                              self.contrastive_loss(pos_id_e, pos_textual_e)
            contrastive_loss = 0.1 * contrastive_loss

        # Regularization
        reg_loss = self.reg_weight * self.reg_loss(user_e, pos_id_e, neg_id_e)

        total_loss = mf_loss + reg_loss + kd_loss + contrastive_loss
        return total_loss

    def full_sort_predict(self, interaction):
        """Prediction with multimodal integration"""
        user = interaction[0]
        
        user_embeddings, _ = self.forward()
        user_e = user_embeddings[user, :]
        
        # Use fused item embeddings for prediction
        if self.use_modality_in_pred:
            all_item_e = self._fuse_embeddings(
                self.i_embedding,
                self.all_visual_e,
                self.all_textual_e
            )
        else:
            all_item_e = self.i_embedding
        
        score = torch.matmul(user_e, all_item_e.transpose(0, 1))
        return score

    def update_item_cache(self):
        """Call this after training to update cached item features"""
        self._precompute_item_features()
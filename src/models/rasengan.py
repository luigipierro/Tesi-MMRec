
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from common.loss import BPRLoss, EmbLoss, L2Loss
from common.abstract_recommender import GeneralRecommender
from torch_geometric.nn import GATConv



class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(AutoEncoder, self).__init__()
        
        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, input_dim)
        )
        
    def forward(self, emb):
        
        # learn at reconstructing the original embeddings
        encoded = self.encoder(emb)
        decoded = self.decoder(encoded)
        return encoded, decoded

class RASENGAN(GeneralRecommender):
    def __init__(self, config, dataset):
        super(RASENGAN, self).__init__(config, dataset)
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.num_users = self.n_users
        self.num_items = self.n_items
        self.hidden_dim = config['embedding_size']
        self.num_heads = config['num_heads']
        self.dropout = config['dropout']
        self.reg_weight = config['reg_weight']
        self.num_layers = config['num_layers']
        self.concat = config['concat']
        self.fusion = config['fusion']

        # list of GAT layers
        self.gat_layers = nn.ModuleList()
        
        # we implement only the first one
        self.gat_layers.append(GATConv(
            in_channels=self.hidden_dim,
            out_channels=self.hidden_dim // self.num_heads,
            heads=self.num_heads,
            dropout=self.dropout,
            add_self_loops=False,
            concat=self.concat
        ))
        
        # the others are optional
        for i in range(1, self.num_layers):
            input_dim = self.hidden_dim if self.concat else self.hidden_dim // self.num_heads
            
            use_concat = self.concat
            if i == self.num_layers - 1:
                use_concat = False

            self.gat_layers.append(GATConv(
                in_channels=input_dim,
                out_channels=self.hidden_dim // self.num_heads,
                heads=self.num_heads,
                dropout=self.dropout,
                add_self_loops=False,
                concat=use_concat
            ))

        # ID embs 
        self.user_embeddings = nn.Embedding(self.num_users, self.hidden_dim)
        self.item_embeddings = nn.Embedding(self.num_items, self.hidden_dim)

        nn.init.xavier_normal_(self.user_embeddings.weight)
        nn.init.xavier_normal_(self.item_embeddings.weight)
        

        # we use MMRec v_feat and t_feat to save user and item pre-trained review features
        self.user_review_embs = torch.tensor(self.v_feat, dtype=torch.float32).to(self.device)
        self.item_review_embs = torch.tensor(self.t_feat, dtype=torch.float32).to(self.device)

        # linear project for fusion item review and id emb
        self.linear_item = nn.Sequential(
            nn.Linear(self.hidden_dim + self.item_review_embs.shape[1], self.hidden_dim),
            nn.ReLU()
        )

        # recommendation and regularization losses 
        self.mf_loss = BPRLoss()
        self.emb_loss = EmbLoss()
        
        # model the IG as a bipartite to work better with GATconv
        self._create_graph_data()
    
    def _create_graph_data(self):
        
        # reshape the interaction matrix as a bipartite graph
        user_indices, item_indices = self.interaction_matrix.nonzero()
        user_item_edges = torch.tensor(np.vstack([
            np.concatenate([user_indices, item_indices + self.num_users]),
            np.concatenate([item_indices + self.num_users, user_indices])
        ]), dtype=torch.long)
        
        self.edge_index = user_item_edges.to(self.device)

    
    def forward(self):

        # get user ID emb
        user_emb = self.user_embeddings.weight

        # early fusion
        if self.fusion == 'early':
            item_emb = self.linear_item(torch.cat([self.item_embeddings.weight, self.item_review_embs], dim=1))
        else:
            item_emb = self.item_embeddings.weight

        x_original = torch.cat([user_emb, item_emb], dim=0)
        x = x_original
        
        # Apply GAT layers
        for i, gat_layer in enumerate(self.gat_layers):
            x_new = gat_layer(x, self.edge_index)
            
            # activation for all but last layer
            if i < len(self.gat_layers) - 1:
                x_new = F.leaky_relu(x_new, negative_slope=0.2)
                x_new = F.dropout(x_new, p=self.dropout, training=self.training)
        
        # get final user and item embs
        user_embeddings, item_embs = torch.split(x, [self.num_users, self.num_items])

        # late fusion
        if self.fusion == 'late':
            item_embeddings = self.linear_item(torch.cat([item_embs, self.item_review_embs], dim=1))
        else:
            item_embeddings = item_embs
        
        return user_embeddings, item_embeddings
    
    def calculate_loss(self, interaction):

        # Get interaction batch
        user, pos_item, neg_item = interaction[0], interaction[1], interaction[2]
        
        # Get embeddings from model
        user_all_embeddings, item_all_embeddings = self.forward()
        
        # Get specific embeddings for this batch
        u_embeddings = user_all_embeddings[user]
        pos_embeddings = item_all_embeddings[pos_item]
        neg_embeddings = item_all_embeddings[neg_item]
        
        # Compute BPR loss
        pos_scores = torch.sum(u_embeddings * pos_embeddings, dim=1)
        neg_scores = torch.sum(u_embeddings * neg_embeddings, dim=1)
        mf_loss = self.mf_loss(pos_scores, neg_scores)
        
        # Get original embeddings for regularization
        user_emb_0 = self.user_embeddings.weight[user]
        pos_item_emb_0 = self.item_embeddings.weight[pos_item]
        neg_item_emb_0 = self.item_embeddings.weight[neg_item]
        
        # # compute MSE if autoencoder, regularization otherwise
        # side_loss = 0
        # if self.pt_modality == 'autoencoder':
        #     _, _, ae_loss = self.encode_embeddings()
        #     side_loss = self.ae_weight * ae_loss
        # elif self.pt_modality == 'concat':
        #     reg_loss = self.emb_loss(user_emb_0, pos_item_emb_0, neg_item_emb_0,
        #             # self.user_concat(torch.cat([user_emb_0, self.user_pt_linear(self.user_review_embs[user])])),
        #             # self.item_concat(torch.cat([pos_item_emb_0, self.item_pt_linear(self.item_review_embs[pos_item])])),
        #             # self.item_concat(torch.cat([neg_item_emb_0, self.item_pt_linear(self.item_review_embs[neg_item])]))
        #             )
        #     side_loss = self.reg_weight * reg_loss
        # else:
        #     reg_loss = self.emb_loss(user_emb_0, pos_item_emb_0, neg_item_emb_0)
        #     side_loss = self.reg_weight * reg_loss
        
        # side_loss
        side_loss = self.reg_weight * self.emb_loss(user_emb_0, pos_item_emb_0, neg_item_emb_0)

        # Final loss
        loss = mf_loss + side_loss
        
        return loss


    
    def full_sort_predict(self, interaction):
        user = interaction[0]
        user_all_embeddings, item_all_embeddings = self.forward()
        u_embeddings = user_all_embeddings[user]
        scores = torch.matmul(u_embeddings, item_all_embeddings.transpose(0, 1))
        return scores
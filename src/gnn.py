import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, adjacency, node_features):
        if adjacency.is_sparse:
            support = torch.sparse.mm(adjacency, node_features)
        else:
            support = torch.matmul(adjacency, node_features)
        return self.linear(support)


class GNNTransactionClassifier(nn.Module):
    def __init__(self, node_input_dim, node_hidden_dim, node_output_dim, tx_feature_dim):
        super().__init__()
        self.gcn1 = GCNLayer(node_input_dim, node_hidden_dim)
        self.gcn2 = GCNLayer(node_hidden_dim, node_output_dim)
        self.classifier = nn.Sequential(
            nn.Linear(node_output_dim * 2 + tx_feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, adjacency, node_features, tx_features, from_idx, to_idx):
        node_hidden = F.relu(self.gcn1(adjacency, node_features))
        node_emb = F.relu(self.gcn2(adjacency, node_hidden))
        return self.classify_edges(node_emb, tx_features, from_idx, to_idx)

    def get_node_embeddings(self, adjacency, node_features):
        node_hidden = F.relu(self.gcn1(adjacency, node_features))
        return F.relu(self.gcn2(adjacency, node_hidden))

    def classify_edges(self, node_emb, tx_features, from_idx, to_idx):
        from_emb = node_emb[from_idx]
        to_emb = node_emb[to_idx]
        edge_features = torch.cat([from_emb, to_emb, tx_features], dim=1)
        return self.classifier(edge_features).squeeze(1)

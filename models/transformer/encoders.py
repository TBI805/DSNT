from torch.nn import functional as F
from models.transformer.grid_aug import BoxRelationalEmbedding
from models.transformer.utils import PositionWiseFeedForward
import torch
from torch import nn
from models.transformer.attention import MultiHeadBoxAttention as MultiHeadAttention, MultiHeadBoxAttention



class EncoderLayer(nn.Module):
    def __init__(self, d_model=512, d_k=128, d_v=128, h=4, d_ff=2048, dropout=.1, identity_map_reordering=False,
                 attention_module=None, attention_module_kwargs=None):
        super(EncoderLayer, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        self.mhatt = MultiHeadAttention(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
                                                attention_module=attention_module,
                                                attention_module_kwargs=attention_module_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.lnorm1 = nn.LayerNorm(d_model)
        self.lnorm2 = nn.LayerNorm(d_model)
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

    def forward(self, queries, keys, values, relative_geometry_weights, attention_mask=None, attention_weights=None, pos=None):

        grid_pos = pos

        att = self.mhatt(queries, keys, values, grid_pos, relative_geometry_weights, attention_mask, attention_weights)
        att = self.lnorm2(queries + self.dropout(att))
        ff = self.pwff(att)

        return ff

class MultiLevelEncoder(nn.Module):
    def __init__(self, N, padding_idx, d_model=512, d_k=128, d_v=128, h=4, d_ff=2048, dropout=.1,
                 identity_map_reordering=False, attention_module=None, attention_module_kwargs=None):
        super(MultiLevelEncoder, self).__init__()
        self.d_model = d_model
        self.dropout = dropout
        self.layers = nn.ModuleList([EncoderLayer(d_model, d_k, d_v, h, d_ff, dropout,
                                                  identity_map_reordering=identity_map_reordering,
                                                  attention_module=attention_module,
                                                  attention_module_kwargs=attention_module_kwargs)
                                     for _ in range(N)])
        self.padding_idx = padding_idx

        self.WGs = nn.ModuleList([nn.Linear(64, 1, bias=True) for _ in range(h)])

    def forward(self, grids, attention_weights=None, pos=None):
        # input (b_s, seq_len, d_in)
        attention_mask = (torch.sum(grids, -1) == self.padding_idx).unsqueeze(1).unsqueeze(1)  # (b_s, 1, 1, seq_len)

        # grid geometry embedding
        # follow implementation of https://github.com/yahoo/object_relation_transformer/blob/ec4a29904035e4b3030a9447d14c323b4f321191/models/RelationTransformerModel.py
        relative_geometry_embeddings = BoxRelationalEmbedding(grids)

        flatten_relative_geometry_embeddings = relative_geometry_embeddings.view(-1, 64)
        box_size_per_head = list(relative_geometry_embeddings.shape[:3])
        box_size_per_head.insert(1, 1)
        relative_geometry_weights_per_head = \
            [layer(flatten_relative_geometry_embeddings).view(box_size_per_head) for layer in self.WGs]
        relative_geometry_weights = torch.cat((relative_geometry_weights_per_head), 1)
        relative_geometry_weights = F.relu(relative_geometry_weights)

        out = grids
        for layer in self.layers:
            out = layer(out, out, out, relative_geometry_weights, attention_mask, attention_weights, pos=pos)

        return out, attention_mask


class TransformerEncoder(MultiLevelEncoder):
    def __init__(self, N, padding_idx, d_in=2048, **kwargs):
        super(TransformerEncoder, self).__init__(N, padding_idx, **kwargs)
        self.fc = nn.Linear(d_in, self.d_model)
        self.dropout = nn.Dropout(p=self.dropout)
        self.layer_norm = nn.LayerNorm(self.d_model)

    def forward(self, grids, attention_weights=None, pos=None):
        mask_grids = (torch.sum(grids, dim=-1) == 0).unsqueeze(-1)
        out_grid = F.relu(self.fc(grids))
        out_grid = self.dropout(out_grid)
        out_grid = self.layer_norm(out_grid)
        out_grid = out_grid.masked_fill(mask_grids, 0)
        return super(TransformerEncoder, self).forward(out_grid,
                                                       attention_weights=attention_weights,
                                                       pos=pos)


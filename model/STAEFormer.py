import torch.nn as nn
import torch
from torchinfo import summary
import numpy as np


class AttentionLayer(nn.Module):
    """Perform attention across the -2 dim (the -1 dim is `model_dim`).

    Make sure the tensor is permuted to correct shape before attention.

    E.g.
    - Input shape (batch_size, in_steps, num_nodes, model_dim).
    - Then the attention will be performed across the nodes.

    Also, it supports different src and tgt length.

    But must `src length == K length == V length`.

    """

    def __init__(self, model_dim, num_heads=8, mask=False):
        super().__init__()

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.mask = mask

        self.head_dim = model_dim // num_heads

        self.FC_Q = nn.Linear(model_dim, model_dim)
        self.FC_K = nn.Linear(model_dim, model_dim)
        self.FC_V = nn.Linear(model_dim, model_dim)

        self.out_proj = nn.Linear(model_dim, model_dim)

    def forward(self, query, key, value):
        # Q    (batch_size, ..., tgt_length, model_dim)
        # K, V (batch_size, ..., src_length, model_dim)
        batch_size = query.shape[0]
        tgt_length = query.shape[-2]
        src_length = key.shape[-2]

        query = self.FC_Q(query)
        key = self.FC_K(key)
        value = self.FC_V(value)

        # Qhead, Khead, Vhead (num_heads * batch_size, ..., length, head_dim)
        query = torch.cat(torch.split(query, self.head_dim, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.head_dim, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.head_dim, dim=-1), dim=0)

        key = key.transpose(
            -1, -2
        )  # (num_heads * batch_size, ..., head_dim, src_length)

        attn_score = (
            query @ key
        ) / self.head_dim**0.5  # (num_heads * batch_size, ..., tgt_length, src_length)

        if self.mask:
            mask = torch.ones(
                tgt_length, src_length, dtype=torch.bool, device=query.device
            ).tril()  # lower triangular part of the matrix
            attn_score.masked_fill_(~mask, -torch.inf)  # fill in-place

        attn_score = torch.softmax(attn_score, dim=-1)
        out = attn_score @ value  # (num_heads * batch_size, ..., tgt_length, head_dim)
        out = torch.cat(
            torch.split(out, batch_size, dim=0), dim=-1
        )  # (batch_size, ..., tgt_length, head_dim * num_heads = model_dim)

        out = self.out_proj(out)

        return out


class SelfAttentionLayer(nn.Module):
    def __init__(
        self, model_dim, feed_forward_dim=2048, num_heads=8, dropout=0, mask=False
    ):
        super().__init__()

        self.attn = AttentionLayer(model_dim, num_heads, mask)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, dim=-2):
        x = x.transpose(dim, -2)
        # x: (batch_size, ..., length, model_dim)
        residual = x
        out = self.attn(x, x, x)  # (batch_size, ..., length, model_dim)
        out = self.dropout1(out)
        out = self.ln1(residual + out)

        residual = out
        out = self.feed_forward(out)  # (batch_size, ..., length, model_dim)
        out = self.dropout2(out)
        out = self.ln2(residual + out)

        out = out.transpose(dim, -2)
        return out


class STAEformer(nn.Module):
    def __init__(self, **model_args):
        super().__init__()

        self.num_nodes = model_args['num_channels']
        self.in_steps = model_args['history_seq_len']
        self.out_steps = model_args['future_seq_len']
        self.steps_per_day = 24 * 60 // model_args['time_interval']
        self.input_dim = model_args['input_dim']
        self.output_dim = model_args['output_dim']
        self.input_embedding_dim = model_args['input_embedding_dim']
        self.tod_embedding_dim = model_args['tod_embedding_dim']
        self.dow_embedding_dim = model_args['dow_embedding_dim']
        self.spatial_embedding_dim = model_args['spatial_embedding_dim']
        self.adaptive_embedding_dim = model_args['adaptive_embedding_dim']
        self.model_dim = (
            self.input_embedding_dim
            + self.tod_embedding_dim
            + self.dow_embedding_dim
            + self.spatial_embedding_dim
            + self.adaptive_embedding_dim
        )
        self.num_heads = model_args['nhead']
        self.num_layers = model_args['e_layers']
        self.use_mixed_proj = model_args['use_mixed_proj']
        self.feed_forward_dim = model_args['dim_feedforward']
        self.dropout = model_args['dropout']
        self.r_s = model_args['r_s']
        self.r_t = model_args['r_t']

        self.input_proj = nn.Linear(self.input_dim, self.input_embedding_dim)
        if self.tod_embedding_dim > 0:
            self.tod_embedding = nn.Embedding(self.steps_per_day, self.tod_embedding_dim)
        if self.dow_embedding_dim > 0:
            self.dow_embedding = nn.Embedding(7, self.dow_embedding_dim)
        if self.spatial_embedding_dim > 0:
            # self.node_emb = nn.Parameter(
            #     torch.empty(self.num_nodes, self.spatial_embedding_dim)
            # )
            self.node_emb = nn.Parameter(
                torch.empty(1, 1, self.spatial_embedding_dim)
            )
            nn.init.xavier_uniform_(self.node_emb)
        if self.adaptive_embedding_dim > 0:
            # self.adaptive_embedding = nn.init.xavier_uniform_(
            #     nn.Parameter(torch.empty(self.in_steps, self.num_nodes, self.adaptive_embedding_dim))
            # )
            self.adaptive_embedding = nn.init.xavier_uniform_(
                nn.Parameter(torch.empty(self.in_steps, 1, self.adaptive_embedding_dim))
            )

        if self.use_mixed_proj:
            self.output_proj = nn.Linear(
                self.in_steps * self.model_dim, self.out_steps * self.output_dim
            )
        else:
            self.temporal_proj = nn.Linear(self.in_steps, self.out_steps)
            self.output_proj = nn.Linear(self.model_dim, self.output_dim)

        self.output_proj2 = nn.Linear(
                self.in_steps * self.model_dim, self.model_dim
            )

        self.attn_layers_t = nn.ModuleList(
            [
                SelfAttentionLayer(self.model_dim, self.feed_forward_dim, self.num_heads, self.dropout)
                for _ in range(self.num_layers)
            ]
        )

        self.attn_layers_s = nn.ModuleList(
            [
                SelfAttentionLayer(self.model_dim, self.feed_forward_dim, self.num_heads, self.dropout)
                for _ in range(self.num_layers)
            ]
        )

        self.output_layer = nn.Linear(self.model_dim, self.out_steps)

        # self.W_s = nn.Parameter(torch.randn(self.num_nodes, self.r_s))
        self.W_s = nn.Parameter(torch.randn(1, self.r_s))
        self.W_t = nn.Parameter(torch.randn(self.in_steps, self.r_t))

    def forward(self, x):
        # x: (batch_size, in_steps, num_nodes, input_dim+tod+dow=3)
        batch_size = x.shape[0]

        if self.tod_embedding_dim > 0:
            tod = x[..., 1]
        if self.dow_embedding_dim > 0:
            dow = x[..., 2]
        x = x[..., : self.input_dim]

        x = self.input_proj(x)  # (batch_size, in_steps, num_nodes, input_embedding_dim)
        features = [x]
        if self.tod_embedding_dim > 0:
            tod_emb = self.tod_embedding(
                (tod * self.steps_per_day).long()
            )  # (batch_size, in_steps, num_nodes, tod_embedding_dim)
            features.append(tod_emb)
        if self.dow_embedding_dim > 0:
            dow_emb = self.dow_embedding(
                dow.long()
            )  # (batch_size, in_steps, num_nodes, dow_embedding_dim)
            features.append(dow_emb)
        if self.spatial_embedding_dim > 0:
            # spatial_emb = self.node_emb.expand(
            #     batch_size, self.in_steps, *self.node_emb.shape
            # )
            spatial_emb = self.node_emb.expand(
                batch_size, self.in_steps, self.num_nodes, self.spatial_embedding_dim
            )
            features.append(spatial_emb)
        if self.adaptive_embedding_dim > 0:
            # adp_emb = self.adaptive_embedding.expand(
            #     size=(batch_size, *self.adaptive_embedding.shape)
            # )
            adp_emb = self.adaptive_embedding.expand(
                batch_size,
                self.in_steps,
                self.num_nodes,
                self.adaptive_embedding_dim
            )
            features.append(adp_emb)
        x = torch.cat(features, dim=-1)  # (batch_size, in_steps, num_nodes, model_dim)

        intermediate_features = []

        for attn in self.attn_layers_t:
            x = attn(x, dim=1)
        for attn in self.attn_layers_s:
            x = attn(x, dim=2)
            inter = x.transpose(1, 2)
            inter = inter.reshape(
                batch_size, self.num_nodes, self.in_steps * self.model_dim
            )
            inter = self.output_proj2(inter).view(
                batch_size, self.num_nodes, self.model_dim
            )
            intermediate_features.append(inter)
        # (batch_size, in_steps, num_nodes, model_dim)

        if self.use_mixed_proj:
            out = x.transpose(1, 2)  # (batch_size, num_nodes, in_steps, model_dim)
            out = out.reshape(
                batch_size, self.num_nodes, self.in_steps * self.model_dim
            )
            out = self.output_proj(out).view(
                batch_size, self.num_nodes, self.out_steps, self.output_dim
            )
            out = out.transpose(1, 2)  # (batch_size, out_steps, num_nodes, output_dim)
        else:
            out = x.transpose(1, 3)  # (batch_size, model_dim, num_nodes, in_steps)
            out = self.temporal_proj(
                out
            )  # (batch_size, model_dim, num_nodes, out_steps)
            out = self.output_proj(
                out.transpose(1, 3)
            )  # (batch_size, out_steps, num_nodes, output_dim)

        intermediate = torch.stack(intermediate_features, dim=0)

        out = out.squeeze(-1) 

        intermediate_last = intermediate[-1,:,:,:]
        intermediate_final = self.output_layer(intermediate_last).permute(0, 2, 1)  
        
        Z_l = intermediate_final.mean(dim=1)  
        similarity_s = torch.matmul(Z_l.unsqueeze(2), Z_l.unsqueeze(1))  
        R_s = torch.softmax(similarity_s / np.sqrt(self.model_dim), dim=-1) 
        U_s = torch.matmul(R_s, self.W_s.expand(self.num_nodes, self.r_s)) 

        Z_l2 = intermediate_final.mean(dim=2)  
        similarity_t = torch.matmul(Z_l2.unsqueeze(2), Z_l2.unsqueeze(1)) 
        R_t = torch.softmax(similarity_t / np.sqrt(self.model_dim), dim=-1) 
        U_t = torch.matmul(R_t, self.W_t) 

        return out, intermediate_last, U_s, U_t


if __name__ == "__main__":
    model = STAEformer(207, 12, 12)
    summary(model, [64, 12, 207, 3])
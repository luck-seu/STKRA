import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from einops import repeat

class SequenceAugmentor:
    def __init__(self, noise_std):
        self.noise_std = noise_std  

    def mask(self, sequence, m):
        m = torch.tensor(int(m), dtype=torch.int64, device=sequence.device)
        seq_length = sequence.size()[0]
        indices = torch.randperm(seq_length)[:m].to(sequence.device) 
        sequence[indices] = 0
        return sequence
    
    
    def add_noise(self, sequence, strength=1.0):
        noise_std = 0.05
        noise = torch.randn_like(sequence) * noise_std
        return sequence + noise

    def augment_mask(self, sequences):
        mask_ratio = 0.1
        mask = torch.rand_like(sequences) > mask_ratio
        return sequences * mask
    
    def augment_noise(self, sequences):
        augmented_sequences = []
        max_value = torch.max(sequences)
        for sequence in sequences:
            augmented_sequences.append(self.add_noise(sequence,  max_value))
        return torch.stack(augmented_sequences)
    
class STKRA_student_general(nn.Module):
    def __init__(self, **model_args):
        super(STKRA_student_general, self).__init__()

        self.history_seq_len = model_args['history_seq_len']
        self.future_seq_len = model_args['future_seq_len']
        self.num_channels = model_args['num_channels']
        self.d_t = model_args['d_t']
        self.d_s = model_args['d_s']
        self.d_f = model_args['d_f']
        self.d_model = self.d_t + self.d_s + self.d_f

        self.r_s = model_args['r_s']
        self.r_t = model_args['r_t']
        
        self.emb_dropout = model_args['emb_dropout']
        self.nhead = model_args['nhead']
        self.dim_feedforward = model_args['dim_feedforward']
        self.num_layers = model_args['num_layers']

        self.build()

    def build(self):
        self.te = nn.Parameter(torch.randn(self.num_channels, self.d_t))
        self.se = nn.Parameter(torch.randn(self.num_channels, self.d_s))

        self.W_s = nn.Parameter(torch.randn(self.num_channels, self.r_s))
        self.W_t = nn.Parameter(torch.randn(self.history_seq_len, self.r_t))

        self.layers = nn.ModuleList()

        for _ in range(self.num_layers - 1):
            self.layers.append(nn.Sequential(
                nn.Linear(self.d_model, self.d_model),
                nn.ReLU()
            ))
        self.output_layer = nn.Linear(self.d_model, self.future_seq_len)
        
        self.projection = nn.Linear(self.future_seq_len, self.d_f)

    def forward(self,
                x: torch.Tensor) -> torch.Tensor:

        x = x[..., 0] 
        x = self.projection(x.permute(0, 2, 1)) 
        B, T, N = x.shape

        te = repeat(self.te, 'n d -> b n d', b=B) 
        se = repeat(self.se, 'n d -> b n d', b=B)
        h = torch.cat([x, te, se], dim=-1) 
        
        intermediates = []
        for layer in self.layers:
            h = layer(h)  

            intermediates.append(h)
        
        output = self.output_layer(h).permute(0, 2, 1).squeeze(-1)  
        intermediate = torch.stack(intermediates, dim=0)

        intermediate_last = intermediate[-1,:,:,:]
        intermediate_final = self.output_layer(intermediate_last).permute(0, 2, 1)  
        
        Z_l = intermediate_final.mean(dim=1)  
        similarity_s = torch.matmul(Z_l.unsqueeze(2), Z_l.unsqueeze(1))  
        R_s = torch.softmax(similarity_s / np.sqrt(self.d_model), dim=-1) 
        U_s = torch.matmul(R_s, self.W_s) 

        Z_l2 = intermediate_final.mean(dim=2) 
        similarity_t = torch.matmul(Z_l2.unsqueeze(2), Z_l2.unsqueeze(1))
        R_t = torch.softmax(similarity_t / np.sqrt(self.d_model), dim=-1)  
        U_t = torch.matmul(R_t, self.W_t) 

        return output, intermediate_last, U_s, U_t
    
class STKRA_student_residual(nn.Module):
    def __init__(self, **model_args):
        super(STKRA_student_residual, self).__init__()

        self.history_seq_len = model_args['history_seq_len']
        self.future_seq_len = model_args['future_seq_len']
        self.num_channels = model_args['num_channels']
        self.d_t = model_args['d_t']
        self.d_s = model_args['d_s']
        self.d_f = model_args['d_f']
        self.d_model = self.d_t + self.d_s + self.d_f

        self.num_layers = model_args['num_layers']

        self.num_prototypes = model_args['num_prototypes']

        self.build()

    def build(self):
        self.layers = nn.ModuleList()
        self.layers.append(nn.Sequential(
            nn.Linear(self.history_seq_len, self.d_model),
            nn.ReLU()
        ))
        for _ in range(self.num_layers - 1):
            self.layers.append(nn.Sequential(
                nn.Linear(self.d_model, self.d_model),
                nn.ReLU()
            ))

        self.memory = nn.Parameter(
            torch.randn(self.num_prototypes, self.d_model)
        )
        self.residual_head = nn.Linear(self.d_model, self.future_seq_len, bias=False)

        self.gate_linear = nn.Linear(self.d_model, self.d_model*2)
        self.value_linear = nn.Linear(self.history_seq_len, self.d_model*2)
        self.projection = nn.Linear(self.d_model*2, self.future_seq_len)

    def forward(self, x: torch.Tensor, F_l: torch.tensor) -> torch.Tensor:
        x = x[..., 0]
        B, T, N = x.shape
        
        x_flat = x.reshape(-1, T)
        augmentor = SequenceAugmentor(noise_std=0.05)
        x_noise = augmentor.augment_noise(x_flat).reshape(B, N, T)
        x_mask = augmentor.augment_mask(x_flat).reshape(B, N, T)
        
        h_orig = x.permute(0, 2, 1)
        h_noise = x_noise
        h_mask = x_mask

        intermediates_orig, intermediates_noise, intermediates_mask = [], [], []
        
        for layer in self.layers:
            h_orig = layer(h_orig)
            h_noise = layer(h_noise) 
            h_mask = layer(h_mask)
            
            intermediates_orig.append(h_orig.reshape(B, -1))  
            intermediates_noise.append(h_noise.reshape(B, -1))
            intermediates_mask.append(h_mask.reshape(B, -1))

        g = torch.sigmoid(self.gate_linear(F_l))
        v = self.value_linear(x.permute(0, 2, 1))
        output = g * v
        output = self.projection(output).permute(0, 2, 1).squeeze(-1)

        intermediate_orig = torch.stack(intermediates_orig, dim=0) 
        intermediate_noise = torch.stack(intermediates_noise, dim=0)
        intermediate_mask = torch.stack(intermediates_mask, dim=0)
        
        return output, intermediate_orig, intermediate_noise, intermediate_mask
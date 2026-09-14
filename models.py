import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

class SimCSELoss(nn.Module):
    def __init__(self, temperature=0.05, mode = "merged"):
        super(SimCSELoss, self).__init__()
        self.temperature = temperature
        self.mode = mode

    def forward(self, z1, z2):
        sim_matrix = F.cosine_similarity(z1.unsqueeze(1), z2.unsqueeze(0), dim=-1) / self.temperature
        labels = torch.arange(z1.size(0), device=z1.device)
        loss = F.cross_entropy(sim_matrix, labels)
        return loss

class UnsupervisedDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]["text"]


# (B, N, dim) -> (B, dim)
def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    return sum_embeddings / sum_mask
    

class FrozenExtractorModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        
        self.base_model = AutoModel.from_pretrained(
            model_name, 
            output_hidden_states=True,
            use_safetensors=True
        )
        
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        self.base_model.eval()

    def train(self, mode = True):
        super().train(mode)
        self.base_model.eval()

    def forward(self, input_ids, attention_mask=None, **kwargs):
        with torch.no_grad():
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **kwargs
            )
            
        all_hidden_states = torch.stack(outputs.hidden_states, axis = 0)  # (n_layers, B, N, hidden_dim)
        all_hidden_states = torch.swapaxes(all_hidden_states, 0, 1)   # (B, n_layers, N, hidden_dim)
        return all_hidden_states, outputs.last_hidden_state


class HeadLevelCombination(nn.Module):
    def __init__(self, n_heads, n_layers, hidden_dim):
        super().__init__()
        
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.target_dim = hidden_dim
        self.head_dim = self.target_dim // n_heads
        
        self.q = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, self.target_dim)
        )
        self.k = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, self.target_dim)
        )
        self.v = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, self.target_dim)
        )
        
        self.ffn = nn.Sequential(
            nn.Linear(self.target_dim, self.target_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.target_dim, self.target_dim)
        )
        
        self.norm = nn.LayerNorm(self.target_dim)
        
        self.dropout = nn.Dropout(0.1)
        

    def whitening(self, x, target_dim = None, normalize = True):
        """Applies BERT-Whitening independently per batch and per layer using fully batched PyTorch operations.

        Args:
            x: Input tensor of shape (B, n_layers, N_token, hidden_dim)
            target_dim: Desired output dimension k (k <= hidden_dim). If None, keeps
            hidden_dim.
            normalize: If True, applies L2 normalization along the hidden dimension.

        Returns:
            Whitened tensor of shape (B, n_layers, N_token, target_dim)
        """
        N, D = x.shape[-2::]
        # B, L, N, D = x.shape
        if target_dim is None:
            target_dim = D

        # 1. Compute mean across N_token dimension -> (B, L, 1, D)
        mu = x.mean(dim=2, keepdim=True)

        # 2. Center inputs -> (B, L, N, D)
        x_centered = x - mu

        # 3. Batched covariance matrix calculation -> (B, L, D, D)
        cov = torch.matmul(x_centered.transpose(-2, -1), x_centered) / (N - 1)

        # 4. Batched SVD (S is sorted in descending order)
        # U: (B, L, D, D), S: (B, L, D)
        U, S, _ = torch.linalg.svd(cov)

        # 5. Truncate to target_dim for dimensionality reduction
        U_k = U[..., :target_dim]  # (B, L, D, target_dim)
        S_k = S[..., :target_dim]  # (B, L, target_dim)

        # 6. Compute transformation matrix W = U_k * S_k^(-1/2)
        # Clamp small singular values to prevent division by zero / numerical instability
        scale = torch.rsqrt(torch.clamp(S_k, min=1e-6))  # (B, L, target_dim)
        W = U_k * scale.unsqueeze(-2)  # (B, L, D, target_dim)

        # 7. Apply transformation -> (B, L, N, target_dim)
        x_whitened = torch.matmul(x_centered, W)

        # 8. Optional L2 normalization for Cosine similarity computation
        if normalize:
            x_whitened = torch.nn.functional.normalize(x_whitened, p=2, dim=-1)

        return x_whitened
        
        
    def forward(self, hidden_states, val = False):  # (B, n_layers, N, hidden_dim)

        # Original: 13/09/2026
        # h1, h2 = Q, K ; h3 = V
        B = hidden_states.shape[0]
        N = hidden_states.shape[2]
        hidden_dim = hidden_states.shape[-1]
        
        # hidden_states = self.whitening(hidden_states, self.target_dim)
        
        q = self.q(hidden_states[::, -1, ...])   # (B, N, hidden_dim)
        k = self.k(hidden_states) # (B, n_layers, N, hidden_dim)
        v = self.v(hidden_states)   # (B, n_layers, N, hidden_dim)
        
        k = k.contiguous().view(B, self.n_layers, N, self.n_heads, self.head_dim)
        k = torch.swapaxes(k, 1, 2)  # (B, N, n_layers, n_heads, head_dim)
        k = k.contiguous().view(B, N, self.n_layers * self.n_heads, self.head_dim)
        k = torch.swapaxes(k, 2, 3)  # (B, N, head_dim, n_layers * n_heads)
        
        q = q.contiguous().view(B, N, self.n_heads, self.head_dim)
        
        m = torch.matmul(q, k) / (self.head_dim ** 0.5)   # (B, N, n_heads, n_layers * n_heads)
        m = F.softmax(m, dim = -1)
        m = self.dropout(m)  
        
        v = v.contiguous().view(B, N, self.n_layers, self.n_heads, self.head_dim)
        v = v.contiguous().view(B, N, self.n_layers * self.n_heads, self.head_dim)
        x = torch.matmul(m, v)  #  (B, N, n_heads, head_dim)
        x = x.contiguous().view(B, N, self.n_heads * self.head_dim)  # (B, N, hidden_dim)
        
        x = self.ffn(x)  # (B, N, hidden_dim)
        x = self.norm(x + hidden_states[::, -1, ...])
    
        return x
    
    
class HLCModel(nn.Module):
    def __init__(self, model_name, hidden_dim, n_layers, n_heads = 1):
        super().__init__()
        
        self.n_layers = n_layers
        self.backbone = FrozenExtractorModel(model_name)
        self.hlc = HeadLevelCombination(n_heads, n_layers, hidden_dim)
        
        self.proj_head = nn.Sequential(
            nn.Linear(self.hlc.target_dim, self.hlc.target_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hlc.target_dim, self.hlc.target_dim),
        )
        
    def forward(self, input_ids, val = False, attention_mask=None, **kwargs):
        x, last = self.backbone(input_ids, attention_mask=attention_mask, **kwargs)
        x = x[::, -self.n_layers::, ...]
        x = self.hlc(x, val)
        x = mean_pooling(x, attention_mask)
        if not val:
            x = self.proj_head(x)
        else:    
            x = self.hlc.whitening(x, self.hlc.target_dim)
        return x

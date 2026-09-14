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


class GlowCouplingBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.half_dim = hidden_dim // 2
        self.net = nn.Sequential(
            nn.Linear(self.half_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)  # Predicts scale (s) and shift (t)
        )

    def forward(self, x: torch.Tensor):
        x1, x2 = x[:, :self.half_dim], x[:, self.half_dim:]
        st = self.net(x1)
        shift, log_scale = st[:, :self.half_dim], st[:, self.half_dim:]
        log_scale = torch.clamp(log_scale, -5.0, 5.0)
        scale = torch.exp(log_scale)
        y2 = x2 * scale + shift
        
        z = torch.cat([x1, y2], dim=-1)
        log_det = log_scale.sum(dim=-1)
        return z, log_det

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
        
        self.flow = nn.ModuleList([
            GlowCouplingBlock(self.hidden_dim) for _ in range(n_layers)
        ])
        
    def compute_bert_flow(
        self, 
        hidden_states: torch.Tensor, 
        return_log_det: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Applies Normalizing Flow transformation to map anisotropic representations into 
        an isotropic Gaussian distribution standard space.

        Args:
            hidden_states (torch.Tensor): Shape (B, n_layers, N, hidden_dim)
            return_log_det (bool): Whether to return log-determinant Jacobian for training loss.

        Returns:
            transformed_states (torch.Tensor): Shape (B, n_layers, N, hidden_dim)
            log_det (torch.Tensor, optional): Shape (B, n_layers, N) if return_log_det=True
        """
        # 1. Capture dimensions
        B, n_layers, N, hidden_dim = hidden_states.shape
        
        # 2. Collapse leading dimensions (B, n_layers, N) -> (-1, hidden_dim)
        x_flat = hidden_states.reshape(-1, hidden_dim)
        
        # 3. Apply Flow transformations sequentially
        z = x_flat
        total_log_det = torch.zeros(x_flat.size(0), device=hidden_states.device)
        
        # Assumes self.flow = nn.ModuleList([GlowCouplingBlock(hidden_dim) for _ in range(num_blocks)])
        for block in self.flow:
            z, log_det = block(z)
            total_log_det = total_log_det + log_det

        # 4. Reshape back to original 4D tensor structure (B, n_layers, N, hidden_dim)
        transformed_states = z.reshape(B, n_layers, N, hidden_dim)

        if return_log_det:
            log_det_reshaped = total_log_det.reshape(B, n_layers, N)
            return transformed_states, log_det_reshaped

        return transformed_states
        
        
    def forward(self, hidden_states, val = False):  # (B, n_layers, N, hidden_dim)

        # Original: 13/09/2026
        # h1, h2 = Q, K ; h3 = V
        B = hidden_states.shape[0]
        N = hidden_states.shape[2]
        hidden_dim = hidden_states.shape[-1]
        
        hidden_states = self.compute_bert_flow(hidden_states)
        
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
        return x

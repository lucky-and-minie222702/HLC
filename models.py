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
        self.head_dim = hidden_dim // n_heads
        
        self.q = nn.Sequential(
            nn.Linear(self.head_dim, self.head_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.head_dim, self.head_dim)
        )
        self.k = nn.Sequential(
            nn.Linear(self.head_dim, self.head_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.head_dim, self.head_dim)
        )
        self.v = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.dropout = nn.Dropout(0.1)
        
        self.norm1 = nn.LayerNorm(n_heads)  # after attention
        self.norm2 = nn.LayerNorm(hidden_dim)  # after ffn
        self.norm3 = nn.LayerNorm(hidden_dim)  # fusion
        
    def forward(self, hidden_states, last_hidden_state, use_original = False):  # (B, n_layers, N, hidden_dim)

        # Original: 13/09/2026
        # h1, h2 = Q, K ; h3 = V
        B = hidden_states.shape[0]
        N = hidden_states.shape[2]
        hidden_dim = hidden_states.shape[-1]
        
        h = hidden_states.contiguous().view(B, self.n_layers, N, self.n_heads, self.head_dim)
        h = torch.swapaxes(h, 1, 2)  # (B, N, n_layers, n_heads, head_dim)
        h = h.contiguous().view(B, N, self.n_layers * self.n_heads, self.head_dim)
        
        l = last_hidden_state.contiguous().view(B, N, self.n_heads, self.head_dim)
        
        q = self.q(h)  # (B, N, n_layers * n_heads, head_dim)
        k = self.k(l)  # (B, N, n_heads, head_dim)
        k = torch.swapaxes(k, 2, 3)  # (B, N, head_dim, n_heads)
        
        m = torch.matmul(q, k) / (self.head_dim ** 0.5)   # (B, N, n_layers * n_heads, n_heads)
        m = F.softmax(m, dim = 1)
        m = self.dropout(m)  # (B, N, n_layers * n_heads, n_heads)
        
        h = torch.swapaxes(h, 2, 3)   # (B, N, head_dim, n_layers * n_head)
        x = torch.matmul(h, m)  #  (B, N, head_dim, n_heads)
        
        x = self.norm1(x)  #  (B, N, head_dim, n_heads)
        x = torch.swapaxes(x, 2, 3)  #  (B, N, n_heads, head_dim)
        x = x.contiguous().view(B, N, self.n_heads * self.head_dim)  # (B, N, hidden_dim)
        x = self.v(x)  # (B, N, hidden_dim)
        
        l = l.contiguous().view(B, N, self.n_heads * self.head_dim)
        x = self.norm3(x + l) # (B, N, hidden_dim)

        return x
    
    
class HLCModel(nn.Module):
    def __init__(self, model_name, hidden_dim, n_layers, n_heads = 1):
        super().__init__()
        
        self.n_layers = n_layers
        self.backbone = FrozenExtractorModel(model_name)
        self.hlc = HeadLevelCombination(n_heads, n_layers, hidden_dim)
        
        self.proj_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
    def forward(self, input_ids, val = False, attention_mask=None, **kwargs):
        x, last = self.backbone(input_ids, attention_mask=attention_mask, **kwargs)
        x = x[::, -self.n_layers::, ...]
        x = self.hlc(x, last, val)
        x = mean_pooling(x, attention_mask)
        if not val:
            x = self.proj_head(x)
        return x

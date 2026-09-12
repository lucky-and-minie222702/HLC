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
        # z (B, dim)
        if self.mode == "normal":
            sim_matrix = F.cosine_similarity(z1.unsqueeze(1), z2.unsqueeze(0), dim=-1) / self.temperature
            labels = torch.arange(z1.size(0), device=z1.device)
            loss = F.cross_entropy(sim_matrix, labels)
            return loss

        elif self.mode == "merged":
            z = torch.cat([z1, z2], dim=0)
            sim_matrix = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=-1) / self.temperature
    
            sim_matrix.fill_diagonal_(-1e18)
            
            B = z1.size(0)
            labels = torch.cat([
                torch.arange(B, 2 * B, device=z1.device),
                torch.arange(0, B, device=z1.device)
            ], dim=0)
            
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
            output_hidden_states=True
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
        return all_hidden_states


class HeadLevelCombination(nn.Module):
    def __init__(self, n_heads, n_layers, hidden_dim):
        super().__init__()
        
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // n_heads
        
        self.w1 = nn.Parameter(torch.empty(n_layers, hidden_dim, n_heads))
        self.w2 = nn.Parameter(torch.empty(n_layers, hidden_dim, n_heads))
        self.w3 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        nn.init.xavier_uniform_(self.w1)
        nn.init.xavier_uniform_(self.w2)
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, hidden_states, use_original = False):  # (B, n_layers, N, hidden_dim)

        if use_original:
            # Original: 11/09/2026
            # h1, h2 = Q, K ; h3 = V
            B = hidden_states.shape[0]
            N = hidden_states.shape[2]
            
            h1 = torch.einsum('blni,lih->blnh', hidden_states, self.w1)  # (B, n_layers, N, n_heads)
            h2 = torch.einsum('blni,lih->blnh', hidden_states, self.w2)  # (B, n_layers, N, n_heads)
            h1 = torch.swapaxes(h1, 2, 3)  # (B, n_layers, n_heads, N)
            
            m = torch.matmul(h1, h2)  # (B, n_layers, n_heads, n_heads)
            m = m.contiguous().view(B, self.n_layers * self.n_heads, self.n_heads)
            m = F.softmax(m, dim = 1)
            m = self.dropout(m)
            # m requires shape (B, n_layers * n_heads, n_heads)
            
            x = hidden_states.contiguous().view(B, self.n_layers, N, self.n_heads, self.head_dim)
            x = torch.swapaxes(x, 1, 2)  # (B, N, n_layers, n_heads, head_dim)
            x = x.contiguous().view(B, N, self.n_layers * self.n_heads, self.head_dim)
            x = torch.swapaxes(x, 2, 3)  # (B, N, head_dim, n_layers * n_heads)
            x = x.contiguous().view(B, N * self.head_dim, self.n_layers * self.n_heads)
            
            x = torch.matmul(x, m)  # (B, N * head_dim, n_heads)
            x = x.contiguous().view(B, N, self.head_dim, self.n_heads)
            x = x.contiguous().view(B, N, self.head_dim * self.n_heads)  # (B, N, hidden_dim)
            x = self.w3(x)
            x = self.dropout(x)  # (B, N, hidden_dim)

            return x
        
        
        # claude optimized
        B, L, N, I = hidden_states.shape
        H, Dh = self.n_heads, self.head_dim

        # fuse w1/w2 (same output width) into one einsum
        w12 = torch.cat((self.w1, self.w2), dim=-1)              # (L, I, 2H)
        h1, h2 = torch.einsum('blni,lig->blng', hidden_states, w12).chunk(2, dim=-1)

        m = torch.einsum('blnh,blng->blhg', h1, h2)               # (B, L, H, H), no explicit transpose
        m = m.reshape(B, L * H, H)
        m = F.softmax(m, dim=1)
        m = self.dropout(m)

        # single permute+copy instead of two swapaxes+view chains
        x = hidden_states.view(B, L, N, H, Dh)
        x = x.permute(0, 2, 4, 1, 3).reshape(B, N * Dh, L * H)

        x = torch.matmul(x, m)                                    # (B, N*Dh, H)
        x = x.reshape(B, N, Dh * H)                               # (B, N, hidden_dim)

        x = self.w3(x)
        x = self.dropout(x)

        return x
    
class HLCModel(nn.Module):
    def __init__(self, model_name, hidden_dim, n_layers, n_heads = 1):
        super().__init__()
        
        self.n_layers = n_layers
        self.backbone = FrozenExtractorModel(model_name)
        self.hlc = HeadLevelCombination(n_heads, n_layers, hidden_dim)
        self.out_head = nn.Linear(hidden_dim, hidden_dim, bias = False)
        
    def forward(self, input_ids, attention_mask=None, **kwargs):
        x = self.backbone(input_ids, attention_mask=attention_mask, **kwargs)
        x = x[::, -self.n_layers::, ...]
        x = self.hlc(x)
        x = mean_pooling(x, attention_mask)
        x = self.out_head(x)
        return x


class BaselineModel(nn.Module):
    def __init__(self, model_name, hidden_dim):
        super().__init__()
        
        self.backbone = FrozenExtractorModel(model_name)
        self.out_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
    def forward(self, input_ids, attention_mask=None, **kwargs):
        x = self.backbone(input_ids, attention_mask=attention_mask, **kwargs)
        x = x[::, -1, ...]
        x = mean_pooling(x, attention_mask)
        x = self.out_head(x)
        return x

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
import math

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
    
    
class AffineCouplingLayer(nn.Module):
    def __init__(self, d_model, mask_type):
        super().__init__()
        self.mask_type = mask_type
        
        hidden_dim = d_model * 2
        self.net = nn.Sequential(
            nn.Linear(d_model // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_model)
        )
        
    def forward(self, x):
        # x shape: (B, n_layers, N, hidden_dim)
        d = x.shape[-1] // 2
        
        # Cắt đôi theo chiều cuối cùng (hidden_dim)
        if self.mask_type == 0:
            x1, x2 = x[..., :d], x[..., d:]
        else:
            x1, x2 = x[..., d:], x[..., :d]
            
        st = self.net(x1)
        s, t = st[..., :d], st[..., d:]
        
        s = torch.tanh(s) # Giới hạn s để tránh nổ gradient
        
        y2 = x2 * torch.exp(s) + t
        y1 = x1
        
        if self.mask_type == 0:
            y = torch.cat([y1, y2], dim=-1)
        else:
            y = torch.cat([y2, y1], dim=-1)
            
        # Tính định thức Jacobian dọc theo chiều feature
        log_det_jacobian = torch.sum(s, dim=-1) 
        
        return y, log_det_jacobian

class RealNVP(nn.Module):
    def __init__(self, d_model, n_flow_layers=4):
        super().__init__()
        self.layers = nn.ModuleList([
            AffineCouplingLayer(d_model, mask_type=i % 2) for i in range(n_flow_layers)
        ])
        
    def forward(self, x):
        log_det_sum = 0
        z = x
        for layer in self.layers:
            z, log_det = layer(z)
            log_det_sum += log_det
            
        # Tính Negative Log-Likelihood cho toàn bộ token/layer
        prior_log_prob = -0.5 * torch.sum(z**2 + math.log(2 * math.pi), dim=-1)
        log_prob = prior_log_prob + log_det_sum
        
        # Tính loss bằng cách lấy trung bình trên tất cả các chiều (Batch, n_layers, N)
        loss = -torch.mean(log_prob) 
        return z, loss


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
    def __init__(self, n_heads, n_layers, hidden_dim, n_flow_layers=4):
        super().__init__()

        self.n_heads = n_heads
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.target_dim = hidden_dim
        self.head_dim = self.target_dim // n_heads

        # Khởi tạo mô hình Flow cho hidden_states
        self.flow = RealNVP(d_model=hidden_dim, n_flow_layers=n_flow_layers)

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

    def compute_bert_flow(self, hidden_states):
        # hidden_states shape: (B, n_layers, N, hidden_dim)
        # Đi qua Flow để được không gian đẳng hướng (isotropic) và lấy flow_loss
        z, flow_loss = self.flow(hidden_states)
        return z, flow_loss

    def forward(self, hidden_states, val=False):  
        B = hidden_states.shape[0]
        N = hidden_states.shape[2]
        
        # 1. Thực hiện thao tác compute_bert_flow
        # Nếu đang val (inference), ta vẫn đi qua flow để lấy Z, có thể bỏ qua flow_loss
        hidden_states, flow_loss = self.compute_bert_flow(hidden_states)

        # 2. Xử lý Attention với hidden_states đã được chuẩn hóa (Z)
        q = self.q(hidden_states[:, -1, ...])   # (B, N, hidden_dim)
        k = self.k(hidden_states)               # (B, n_layers, N, hidden_dim)
        v = self.v(hidden_states)               # (B, n_layers, N, hidden_dim)

        k = k.contiguous().view(B, self.n_layers, N, self.n_heads, self.head_dim)
        k = torch.swapaxes(k, 1, 2)             # (B, N, n_layers, n_heads, head_dim)
        k = k.contiguous().view(B, N, self.n_layers * self.n_heads, self.head_dim)
        k = torch.swapaxes(k, 2, 3)             # (B, N, head_dim, n_layers * n_heads)

        q = q.contiguous().view(B, N, self.n_heads, self.head_dim)

        m = torch.matmul(q, k) / (self.head_dim ** 0.5)   
        m = F.softmax(m, dim=-1)
        m = self.dropout(m)  

        v = v.contiguous().view(B, N, self.n_layers, self.n_heads, self.head_dim)
        v = v.contiguous().view(B, N, self.n_layers * self.n_heads, self.head_dim)
        
        x = torch.matmul(m, v)  
        x = x.contiguous().view(B, N, self.n_heads * self.head_dim)  

        x = self.ffn(x)  

        # Trả về output và flow_loss để dùng cho bước huấn luyện
        return x, flow_loss


class HLCModel(nn.Module):
    def __init__(self, model_name, hidden_dim, n_layers, n_heads=1, n_flow_layers=4):
        super().__init__()

        self.n_layers = n_layers
        self.backbone = FrozenExtractorModel(model_name)
        
        # Truyền thêm n_flow_layers
        self.hlc = HeadLevelCombination(n_heads, n_layers, hidden_dim, n_flow_layers)

        self.proj_head = nn.Sequential(
            nn.Linear(self.hlc.target_dim, self.hlc.target_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hlc.target_dim, self.hlc.target_dim),
        )

    def forward(self, input_ids, val=False, attention_mask=None, **kwargs):
        x, last = self.backbone(input_ids, attention_mask=attention_mask, **kwargs)
        
        # Lấy n_layers cuối cùng
        x = x[:, -self.n_layers:, ...]
        
        # Đưa qua HLC và nhận lại output cùng flow_loss
        x, flow_loss = self.hlc(x, val)
        
        # Giả định bạn có hàm mean_pooling được định nghĩa bên ngoài
        x = mean_pooling(x, attention_mask)
        
        if not val:
            x = self.proj_head(x)
            
        # Bắt buộc trả về flow_loss để optimize
        return x, flow_loss
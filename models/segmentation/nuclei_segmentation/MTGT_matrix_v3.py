import os 
import torch
cpu_num = 6
os.environ["OMP_NUM_THREADS"] = str(cpu_num)
os.environ["OPENBLAS_NUM_THREADS"] = str(cpu_num)
os.environ["MKL_NUM_THREADS"] = str(cpu_num)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(cpu_num)
os.environ["NUMEXPR_NUM_THREADS"] = str(cpu_num)
torch.set_num_threads(cpu_num)
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('./multitask')
from .Model.DSMIL import FCLayer, BClassifier
from .Model.func import Attn_Net_Gated, SNN_Block, init_max_weights
# from timm.models.layers import DropPath
from einops import rearrange
import numpy as np
import ot
# from geomloss import SamplesLoss

torch.set_printoptions(precision=8)

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # B x num_heads x N x N

        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)

        return x

class AttentionBlock(nn.Module):
    def __init__(self, dim = 1024, depth = 2, num_heads = 1, dim_head = 1024, mlp_dim = 1024, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, num_heads = num_heads)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class DSMIL_enc_matrixversion(nn.Module):
    def __init__(self, i_classifier, b_classifier, dropout = 0, output_dim = 1024):
        super(DSMIL_enc_matrixversion, self).__init__()
        self.i_classifier = i_classifier
        self.b_classifier = b_classifier

        cls_fc = [  nn.Linear(1024, output_dim), nn.ReLU(),
                nn.Linear(output_dim, output_dim), nn.ReLU(),
                nn.Linear(output_dim, output_dim)]
        self.cls_proj = nn.Sequential(*cls_fc)
        
    def forward(self, **kwargs):
        x = kwargs['data'] # [13665, 1024]
        feats, classes = self.i_classifier(x)
        emb = self.b_classifier(feats, classes) # [1, 256, 1024]
        cls_emb = self.cls_proj(emb).squeeze() # [256, 1024]
        return cls_emb

class CustomAttn(nn.Module):
    def __init__(
            self,
            dim = 1024,
            num_class = 256
            ):
        super().__init__()
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.proj_v = nn.Linear(256, num_class, bias=False)
        self.rescale = nn.Parameter(torch.ones(1, 1))
        self.proj = nn.Linear(dim, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.GELU(),
            nn.Linear(dim, dim, bias=False),
        )

    def forward(self, x_in):
        """
        x_in: [b,h,w,c]
        return out: [b,h,w,c]
        """
       
        n, c = x_in.shape
        x = x_in.reshape(n, c)
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        # q: b,heads,hw,c
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)


        v = self.proj_v(v).transpose(0, 1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        v = F.normalize(v, dim=-1, p=2)
        attn = (k @ q.transpose(-2, -1))   # A = K^T*Q
        attn = attn * self.rescale
        attn = attn.softmax(dim=-1)
        # x = attn @ v   # b,heads,d,hw
        x = v @ attn # b, heads, dim, n
        out_c = self.proj(x)
        out_p = self.pos_emb(v)
        out = out_c + out_p

        return out

class TG_MSA(nn.Module):    # density - guided multi-head self-attention: 
    def __init__(
            self,
            dim,
            dim_head,
            heads,
            device
    ):
        super().__init__()
        self.device = device
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Linear(dim, dim),
            GELU(),
            nn.Linear(dim, dim),
        )
        self.dim = dim

    def forward(self, x_in, task_x):
        """
        x_in: [b,h,w,c]
        return out: [b,h,w,c]
        """

        task_x = task_x.flatten(2).transpose(1, 2)  


        x_in = x_in.permute(0, 2, 3, 1) #b, c, h, w ->b,h,w,c

        b, h,w,c = x_in.shape
        n=h*w
       
        x = x_in.reshape(b, h*w, c)


        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)


        tmp_task_x = task_x.clone().detach()

        #ori
        q, k, v, task_x = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
                                (q_inp, k_inp, v_inp, task_x))

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
                                (q_inp, k_inp, v_inp))
        
        

        #ori
        tmp_v_inp = v_inp.clone().detach()



        guided_emb_list = []

        task_x = task_x.to(self.device)
        v = v.to(self.device)

        for i in range(b):
            tmp_task_x1 = tmp_task_x[i].clone().detach()
            tmp_v_inp1 = tmp_v_inp[i].clone().detach()


            M = ot.dist(tmp_task_x1.squeeze().cpu().numpy(), tmp_v_inp1.squeeze().cpu().numpy())  # OT中衡量两个矩阵之间的距离

            mean_value = np.mean(M)
            std_deviation = np.std(M)
            M = (M - mean_value) / std_deviation

            uni_a, uni_b = np.ones((n,)) / n, np.ones((n,)) / n
            Ges = torch.tensor(ot.emd(uni_a, uni_b, M), dtype=torch.float32).to(self.device).unsqueeze(0).unsqueeze(0)  # 根据距离返回OT中传输矩阵

            #print('xxx'+'*'*1000,Ges.device,task_x[i].device,v[i].device)

            

            guided_emb = Ges @ task_x[i] * v[i]
            guided_emb_list.append(guided_emb)

        guided_emb = torch.stack(guided_emb_list, dim=0)
        #print('guided_emb1'+'*'*1000,guided_emb.shape)  # torch.Size([4, 1, 8, 1024, 64])
        guided_emb=guided_emb.squeeze()
        #print('guided_emb2'+'*'*1000,guided_emb.shape)  #torch.Size([4, 8, 1024, 64])

        # q: b,heads,hw,c
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        guided_emb = guided_emb.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attn = (k @ q.transpose(-2, -1))   # A = K^T*Q
        attn = attn * self.rescale
        attn = attn.softmax(dim=-1)
        # x = attn @ v   # b,heads,d,hw


        attn = attn.to(self.device)
        


        x = attn @ guided_emb # b, heads, dim, n


        x = x.permute(0, 3, 1, 2)    # Transpose  b, n, heads, dim 

        x = rearrange(x, 'b n h d -> b n (h d)')

        #x = x.to(self.device)
        #self.proj = self.proj.to(device)

        x=x.to(self.proj.weight.device)


        out_c = self.proj(x).view(b, n, c)
        out_p = self.pos_emb(v_inp)


        out = out_c + out_p

        out = out.view(out.size(0), h, w, -1)


        out = out.permute(0, 3, 1, 2) 

        return out

class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


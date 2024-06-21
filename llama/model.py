# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import fairscale.nn.model_parallel.initialize as fs_init
import torch
import torch.nn.functional as F
from fairscale.nn.model_parallel.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
    RowParallelLinear,
)
from torch import nn


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2，保证FFN隐藏层维度是multiple_of 的整数倍
    ffn_dim_multiplier: Optional[float] = None  
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 2048


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps # ε
        self.weight = nn.Parameter(torch.ones(dim)) #可学习参数γ

    def _norm(self, x):
        # RMSNorm
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # 将潜在的半精度浮点数转化为单精度浮点数进行计算。
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    # 计算词向量元素两两分组以后，每组元素对应的旋转角度 
    # arange生成[0,2,4...126]
    # theta = 10000^{-2k/d} 
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # t = [0,....end]
    t = torch.arange(end, device=freqs.device)  # type: ignore
    # t为列向量 freqs为行向量做外积
    # freqs.shape = (t.len(),freqs.len()) #shape (end,dim//2)
    freqs = torch.outer(t, freqs).float()  # type: ignore
    # 生成复数向量
    # torch.polar(abs,angle) -> abs*cos(angle) + abs*sin(angle)*j
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    # freqs_cis.shape  = (end,dim//2)
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    # ndim为x的维度数 ,此时应该为4
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    # (1,x.shape[1],1,x.shape[-1])
    return freqs_cis.view(*shape)

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # xq.shape = [bsz, seqlen, self.n_local_heads, self.head_dim]
    # xq_.shape = [bsz, seqlen, self.n_local_heads, self.head_dim//2 , 2]
    # torch.view_as_complex用于将二维向量转换为复数域 torch.view_as_complex即([x,y]) -> (x+yj)
    # 所以经过view_as_complex变换后xq_.shape = [bsz, seqlen, self.n_local_heads, self.head_dim//2]
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_) # freqs_cis.shape = (1,x.shape[1],1,x.shape[-1])
    
    # xq_ 与freqs_cis广播哈达玛积
    # [bsz, seqlen, self.n_local_heads, self.head_dim//2] * [1,seqlen,1,self.head_dim//2]
    # torch.view_as_real用于将复数再转换回实数向量, 再经过flatten展平第4个维度 
    # [bsz, seqlen, self.n_local_heads, self.head_dim//2] ->[bsz, seqlen, self.n_local_heads, self.head_dim//2,2 ] ->[bsz, seqlen, self.n_local_heads, self.head_dim]
    # 复数xq和复数e^{im\theta}相乘
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    # 根据n_rep，拓展KV
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        # 使用 GQA，一共有 n_kv_heads 组 kv
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        model_parallel_size = fs_init.get_model_parallel_world_size()
        self.n_local_heads = args.n_heads // model_parallel_size  #Q的头数
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size #KV的头数
        self.n_rep = self.n_local_heads // self.n_local_kv_heads  # 一共有 head 个头，kv头有g组，同一个组里面需要共享kv头
        self.head_dim = args.dim // args.n_heads

        self.wq = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim, # Q的头数* head_dim
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )  # [dim, head_dim * dim]
        self.wk = ColumnParallelLinear(
            args.dim,
            self.n_kv_heads * self.head_dim, # K的头数* head_dim
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )  # [dim, head_dim * g]
        self.wv = ColumnParallelLinear(
            args.dim,
            self.n_kv_heads * self.head_dim, # V的头数* head_dim
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )  # [dim, group * head_dim]
        self.wo = RowParallelLinear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False,
            input_is_parallel=True,
            init_method=lambda x: x,
        )  # [head_dim * head, dim]

        self.cache_k = torch.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_local_kv_heads, #KV的头数
                self.head_dim,
            )
        ).cuda()  # [batch, max_seqlen, group, head_dim]
        self.cache_v = torch.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_local_kv_heads, #KV的头数
                self.head_dim,
            )
        ).cuda()  # [batch, max_seqlen, group, head_dim]

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        bsz, seqlen, _ = x.shape  # [batch, seqlen, dim]
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)  
        # xq: [batch, seqlen, head * head_dim]
        # xk, xv: [batch, seqlen, group * head_dim]

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)  # [batch, seqlen, head, head_dim]
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)  # [batch, seqlen, group, head_dim]
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)  # [batch, seqlen, group, head_dim]

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis) #嵌入RoPE位置编码
        
        # 将 cache_k 转化为和 xq 相同的设备和数据类型
        self.cache_k = self.cache_k.to(xq)
        self.cache_v = self.cache_v.to(xq)
        
        # 按此时序列的句子长度把kv添加到cache中
        # 初始在prompt阶段seqlen>=1, 后续生成过程中seqlen==1
        self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
        self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv
        
        # 读取完整的kv
        keys = self.cache_k[:bsz, : start_pos + seqlen]  # [batch, all_seqlen, group, head_dim]
        values = self.cache_v[:bsz, : start_pos + seqlen]  # [batch, all_seqlen, group, head_dim]

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(keys, self.n_rep)  # [batch, all_seqlen, head, head_dim]
        values = repeat_kv(values, self.n_rep)  # [batch, all_seqlen, head, head_dim)

        xq = xq.transpose(1, 2)  # [batch, head, seqlen, head_dim]
        keys = keys.transpose(1, 2)  # [batch, head, all_seqlen, head_dim]
        values = values.transpose(1, 2)  # [batch, head, all_seqlen, head_dim]
        #计算q*k^T / sqrt(d_k)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)  # [batch, head, seqlen, all_seqlen]
        if mask is not None:
            #加入mask，使得前面的token在于后面的token计算attention时得分为0，mask掉
            scores = scores + mask  # [batch, head, all_seqlen, all_seqlen]
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)  # [batch, head, seqlen, all_seqlen]
        output = torch.matmul(scores, values)  # [batch, head, seqlen, head_dim]
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)  # [batch, seqlen, dim]
        return self.wo(output)  # [batch, seqlen, dim]


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()
        # 保证使用SwiGLU后参数量是相同的
        # 2 * dim * hidden_dim = 3 * dim * k * hidden_dim
        # k = 2 / 3

        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        
        # 保证 hidden_dim 是 multiple_of 的整数倍
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        # Linear 1
        self.w1 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        )
        # Linear 2
        self.w2 = RowParallelLinear(
            hidden_dim, dim, bias=False, input_is_parallel=True, init_method=lambda x: x
        )
        # Linear 3
        self.w3 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        )

    def forward(self, x):
        # 使用SwiGLU激活函数
        # f(x) = w2 * (Swish(w1 * x) * w3 * x)
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        # x: [batch, seqlen, dim]
        # 初始版本的tranformer使用PostNorm, Norm(x + f(x)) 
        # 这里使用PreNorm, x + f(Norm(x))
        # 使用 PreNorm，先归一化再输入到函数
        h = x + self.attention.forward(
            self.attention_norm(x), start_pos, freqs_cis, mask
        )  # h: [batch, seqlen, dim]
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size # 词汇表长度
        self.n_layers = params.n_layers # Transformer 层数

        self.tok_embeddings = ParallelEmbedding(
            params.vocab_size, params.dim, init_method=lambda x: x
        ) # token 嵌入层

        
        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params)) 
        # 添加若干层TransformerBlock

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        # 使用RMSorm进行层归一化

        # 输出，将 embedding 转化为 token
        self.output = ColumnParallelLinear(
            params.dim, params.vocab_size, bias=False, init_method=lambda x: x
        )

        # 计算RoPE位置编码，e^{im\theta}, [seq_len, head_dim // 2]
        self.freqs_cis = precompute_freqs_cis(
            self.params.dim // self.params.n_heads, self.params.max_seq_len * 2
        )

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)  # [batch, seqlen, dim]
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        # 生成decoder结构下的单向注意力掩码矩阵，对于第i行，[0, i]位置数值为0, [i+1, seqlen - 1] 位置数值为 -inf
        if seqlen > 1:
            # 首先生成一个[1,1,seqlen,seqlen] 值全为-inf的矩阵块
            mask = torch.full(
                (1, 1, seqlen, seqlen), float("-inf"), device=tokens.device
            )
            # 从diagonal列开始往后生成下三角为0的矩阵
            # torch.triu 返回输入矩阵从diagonal开始的上三角矩阵，其他位置填充0
            mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)

        # 通过 n_layers 次 transformer 的计算
        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        
        # 使用PreNorm，所以在输出前也要归一化一下
        h = self.norm(h) # [batch, seqlen, dim]

        # 将embedding 转化为 token
        output = self.output(h).float() # [batch, seqlen, vacab_size]
        return output

if __name__ == "__main__":
    from fairscale.nn.model_parallel.initialize import (
        get_model_parallel_rank,
        initialize_model_parallel,
        model_parallel_is_initialized,
    )
    import os
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group("nccl")
    if not model_parallel_is_initialized():
        model_parallel_size = int(os.environ.get("WORLD_SIZE", 1))
        initialize_model_parallel(model_parallel_size)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(1)

    model_args: ModelArgs = ModelArgs(
        max_seq_len=128,
        max_batch_size=4,
        dim=4096,
        multiple_of=256,
        n_heads=32,
        n_layers=32, 
        norm_eps=1e-05,
        vocab_size=100
    )
    model = Transformer(model_args)

    x = torch.ones(4, 60, dtype=int)
    model(x, 0)
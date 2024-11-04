import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import warnings

warnings.filterwarnings("ignore")

from config import model_config


def precompute_theta_pos_freqs(head_dim: int, seq_len: int, theta: float = 10000.0):
    assert head_dim % 2 == 0, "dim must be even"

    # theta_i = theta ^ (2i / dim), i -. [0, dim/2]
    theta_numerator = torch.arange(0, head_dim, 2).float()

    theta_i = 1.0 / (theta ** (theta_numerator / head_dim))
    m = torch.arange(seq_len)

    freqs = torch.outer(m, theta_i).float()
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex


def apply_rotary_embeddings(x: torch.Tensor, freqs_complex: torch.Tensor):
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
    x_rotated = x_complex * freqs_complex
    x_out = torch.view_as_real(x_rotated)
    x_out = x_out.reshape(*x.shape)
    return x_out.type_as(x)


def repeat_kv(x: torch.Tensor, n_rep: int):
    batch_size, seq_length, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    x = x.view(batch_size, seq_length, n_kv_heads, 1, head_dim)
    x = x.tile(1, 1, 1, n_rep, 1)
    x = x.view(batch_size, seq_length, n_kv_heads * n_rep, head_dim)
    return x


class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.n_heads = config.n_heads
        self.n_q_heads = config.n_heads
        self.n_kv_heads = (
            config.n_kv_heads if config.n_kv_heads is not None else config.n_heads
        )
        self.n_rep = self.n_q_heads // self.n_kv_heads

        self.head_dim = config.dim // self.n_heads

        self.wq = nn.Linear(config.dim, self.n_q_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(config.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_dim, config.dim, bias=False)

        self.cache_k = torch.zeros(
            (config.batch_size, config.seq_len, self.n_kv_heads, self.head_dim)
        )
        self.cache_v = torch.zeros(
            (config.batch_size, config.seq_len, self.n_kv_heads, self.head_dim)
        )

    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        batch_size, seq_length, _ = x.shape

        xq = self.wq(x).view(batch_size, seq_length, self.n_q_heads, self.head_dim)
        xk = self.wk(x).view(batch_size, seq_length, self.n_kv_heads, self.head_dim)
        xv = self.wv(x).view(batch_size, seq_length, self.n_kv_heads, self.head_dim)

        xq = apply_rotary_embeddings(xq, freqs_complex)
        xk = apply_rotary_embeddings(xk, freqs_complex)

        self.cache_k[:batch_size, start_pos : start_pos + seq_length] = xk
        self.cache_v[:batch_size, start_pos : start_pos + seq_length] = xv

        keys = self.cache_k[:batch_size, : start_pos + seq_length]
        values = self.cache_v[:batch_size, : start_pos + seq_length]

        keys = repeat_kv(keys, self.n_rep)
        values = repeat_kv(values, self.n_rep)

        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        scores = torch.matmul(xq, keys.mT) / math.sqrt(self.head_dim)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)

        output = torch.matmul(scores, values)

        output = (
            output.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_length, -1)
        )

        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()

        hidden_dim = config.dim * 4
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = config.multiple_of * (
            (hidden_dim + config.multiple_of - 1) // config.multiple_of
        )

        self.w1 = nn.Linear(config.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, config.dim, bias=False)
        self.w3 = nn.Linear(config.dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class EncoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.attention = SelfAttention(config)
        self.feed_forward = FeedForward(config)

        self.attention_norm = RMSNorm(config.dim, config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)

    def forward(self, x: torch.Tensor, start_pos: int):
        h = x + self.attention(self.attention_norm(x), start_pos)
        output = h + self.feed_forward(self.ffn_norm(h))
        return output


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        return self.weight * self._norm(x.float()).type_as(x)


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()

        assert config.vocab_size != -1, "Vocab size must be set"

        self.config = config

        self.n_layers = config.n_layers
        self.vocab_size = config.vocab_size
        self.dim = config.dim
        self.tok_embeddings = nn.Embedding(self.vocab_size, self.dim)

        self.layers = nn.ModuleList(
            [EncoderBlock(self.config) for _ in range(self.n_layers)]
        )

        self.norm = RMSNorm(self.dim, config.norm_eps)
        self.output = nn.Linear(self.dim, self.vocab_size, bias=False)

        self.freqs_complex = precompute_theta_pos_freqs(self.dim, config.seq_len)

    def forward(self, x: torch.Tensor, start_pos: int):
        batch_size, seq_length = x.shape

        assert seq_length == 1, "One token at a time"

        h = self.tok_embeddings(x)

        freqs_complex = self.freq_complex[start_pos : start_pos + seq_length]

        for layer in self.layers:
            h = layer(h, start_pos, freqs_complex)

        h = self.norm(h)
        output = self.output(h)
        return output

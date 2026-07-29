import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from dataclasses import dataclass

@dataclass
class GPTConfig:
    vocab_size: int = 8013
    n_embeddings: int = 128
    n_heads: int = 4
    n_layers: int = 4
    block_size: int = 256
    dropout: float = 0.2

class Head(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.head_size = config.n_embeddings // config.n_heads
        self.key = nn.Linear(config.n_embeddings, self.head_size)
        self.query = nn.Linear(config.n_embeddings, self.head_size)
        self.value = nn.Linear(config.n_embeddings, self.head_size)
        self.register_buffer('tril', torch.tril(torch.ones(config.block_size, config.block_size)))
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, attention_mask=None):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        w = q @ k.transpose(-2, -1) / math.sqrt(self.head_size)
        w = w.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        w = F.softmax(w, dim=-1)
        w = self.dropout(w)
        return w @ v
    
class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = nn.ModuleList(Head(config) for head in range(config.n_heads))
        self.projection = nn.Linear(config.n_embeddings, config.n_embeddings)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.projection(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embeddings, 4 * config.n_embeddings),
            nn.ReLU(), 
            nn.Linear(4 * config.n_embeddings, config.n_embeddings),
            nn.Dropout(config.dropout)
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.n_embeddings)
        self.ln2 = nn.LayerNorm(config.n_embeddings)

    def forward(self, x):
        x = x + self.self_attention(self.ln1(x))
        x = x + self.feed_forward(self.ln2(x))
        return x

class ReviewGenerator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embeddings)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embeddings)
        self.blocks = nn.Sequential(*[Block(config) for layer in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.n_embeddings)
        self.lm_head = nn.Linear(config.n_embeddings, config.vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
        return logits, loss

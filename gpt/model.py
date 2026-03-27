import math
import torch
import torch.nn as nn 


class GELU(nn.Module):
    def forward(self, x):
        # This is the exact formula OpenAI used in GPT-2.
        return (
            0.5 * x * (
                1.0 + torch.tanh(
                    math.sqrt(2.0 / math.pi)
                    * (x + 0.044715 * torch.pow(x, 3.0))
                )
            )
        )

class LayerNorm(nn.Module):

    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        # Learnable parameters,both start as 1 and 0 respectively.
        self.weight = nn.Parameter(torch.ones(emb_dim))   # scale γ
        self.bias   = nn.Parameter(torch.zeros(emb_dim))  # shift β

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)          # mean over last dim
        var  = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight * x_norm + self.bias      # rescale & shift


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),  # expand
            GELU(),                                           # activate
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),  # compress back
        )

    def forward(self, x):
        return self.layers(x)


class MultiHeadCausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg["emb_dim"] % cfg["n_heads"] == 0, \
            "emb_dim must be divisible by n_heads"

        self.n_heads   = cfg["n_heads"]
        self.emb_dim   = cfg["emb_dim"]
        self.head_dim  = cfg["emb_dim"] // cfg["n_heads"]  # dim per head

        # Single linear that projects input into Q, K, V all at once (3x)
        self.c_attn = nn.Linear(cfg["emb_dim"], 3 * cfg["emb_dim"], bias=True)
        # Output projection that merges all heads back
        self.c_proj = nn.Linear(cfg["emb_dim"], cfg["emb_dim"], bias=True)

        # Dropout for regularisation
        self.attn_dropout = nn.Dropout(cfg["drop_rate"])
        self.proj_dropout = nn.Dropout(cfg["drop_rate"])

        # Causal mask, registered as a buffer (not a learnable parameter)
        # It's a lower-triangular matrix of ones.
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(cfg["context_length"], cfg["context_length"]))
            .unsqueeze(0).unsqueeze(0)  # shape: [1, 1, T, T]
        )

    def forward(self, x):
        B, T, C = x.shape  # Batch, Sequence-length, Channels (emb_dim)

        # 1. Project input to Q, K, V
        qkv = self.c_attn(x)              # [B, T, 3*C]
        q, k, v = qkv.split(self.emb_dim, dim=2)  # each [B, T, C]

        # 2. Reshape for multi-head: [B, n_heads, T, head_dim]
        def split_heads(t):
            return t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)

        #3. Scaled dot-product attention
        scale  = math.sqrt(self.head_dim)
        scores = (q @ k.transpose(-2, -1)) / scale  # [B, n_heads, T, T]

        # Apply causal mask: set future positions to -inf so softmax → 0
        scores = scores.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        attn   = torch.softmax(scores, dim=-1)
        attn   = self.attn_dropout(attn)

        # 4. Weighted sum of values
        out = attn @ v                             # [B, n_heads, T, head_dim]
        out = out.transpose(1, 2).contiguous()     # [B, T, n_heads, head_dim]
        out = out.view(B, T, C)                    # merge heads → [B, T, C]

        #5. Final output projection
        out = self.proj_dropout(self.c_proj(out))
        return out




class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1   = LayerNorm(cfg["emb_dim"])
        self.attn    = MultiHeadCausalSelfAttention(cfg)
        self.norm2   = LayerNorm(cfg["emb_dim"])
        self.ff      = FeedForward(cfg)

    def forward(self, x):
        # Attention sub-layer with residual
        x = x + self.attn(self.norm1(x))
        # Feed-forward sub-layer with residual
        x = x + self.ff(self.norm2(x))
        return x




class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        #Token embedding maps each token ID to dense vector
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])

        #Position embeddings: tells the model WHERE each token is
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])

        #Dropout after embedding
        self.emb_drop = nn.Dropout(cfg["drop_rate"])

        #Stack of Transformer blocks
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        #Final layer normalisation before the language-model head
        self.final_norm = LayerNorm(cfg["emb_dim"])

        #Language model head maps embeddingsb to vocabulary logits
        # No bias, and we tie weights with token embedding (weight tying)
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

        # Weight tying the output head shares weights with the token embedding.
        # This reduces parameters and often improves performance.
        self.out_head.weight = self.tok_emb.weight

        #Parameter initialisation (matches GPT-2 paper)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialise linear and embedding layers the way GPT-2 does."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        B, T = idx.shape
        assert T <= self.pos_emb.num_embeddings, \
            f"Sequence length {T} exceeds model's context_length"

        #1. Embeddings
        tok_embeds = self.tok_emb(idx)                          # [B, T, C]
        pos_ids    = torch.arange(T, device=idx.device)
        pos_embeds = self.pos_emb(pos_ids)                      # [T, C]
        x          = self.emb_drop(tok_embeds + pos_embeds)     # [B, T, C]

        #2. Transformer blocks
        x = self.trf_blocks(x)                                  # [B, T, C]

        #3. Final norm + head
        x      = self.final_norm(x)
        logits = self.out_head(x)                               # [B, T, V]
        return logits


GPT2_124M_CONFIG = {
    "vocab_size"     : 50257,   # GPT-2's BPE vocabulary size
    "context_length" : 1024,    # maximum tokens the model can "see" at once
    "emb_dim"        : 768,     # each token is represented as a 768-dim vector
    "n_heads"        : 12,      # 12 attention heads
    "n_layers"       : 12,      # 12 stacked Transformer blocks
    "drop_rate"      : 0.0,     # 0 = disabled (best for inference / fine-tuning)
    "qkv_bias"       : True,    # GPT-2 uses bias in Q, K, V projections
}


def build_gpt2_model(config=None):
    cfg = GPT2_124M_CONFIG.copy()
    if config:
        cfg.update(config)

    model = GPTModel(cfg)

    # Print a quick summary of the model size
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[GPTModel] Built GPT-2 with {total_params:,} parameters "
          f"({total_params / 1e6:.1f}M)")
    return model, cfg
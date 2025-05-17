"""
Much of this code is adapted from Andrej Karpathy's NanoGPT
(https://github.com/karpathy/nanoGPT)
"""
from dataclasses import dataclass
import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Optional

from .model import GPT, GPTConfig, MLP

try:
    from spas_sage_attn import spas_sage2_attn_meansim_cuda as _sparge_attn
except ImportError:
    _sparge_attn = None

try:
    from sageattention import sageattn as _sage_direct
    from sageattention import sageattn_varlen as _sage_varlen
except ImportError:
    _sage_direct = None
    _sage_varlen = None


class NonCausalSelfAttention(nn.Module):
    """Bark‑style MHA that supports a (B, T) pad mask and uses Sage/Sparge when available."""

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # projections
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # hyper‑params
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        # feature flags
        self.use_sparge = _sparge_attn is not None and torch.cuda.is_available()
        self.use_sage = (
            (not self.use_sparge) and _sage_direct is not None and torch.cuda.is_available()
        )

    # ---------------------------------------------------------------------
    def _pack_and_sage(self, q, k, v, pad_mask):
        """Pack variable‑length batch for sageattn_varlen, call it, then scatter back."""
        B, H, T, D = q.shape
        lengths = pad_mask.sum(dim=1).to(torch.int32)  # (B,)
        max_len = int(lengths.max())

        # build cumulative sequence lengths (B+1,)
        cu = torch.zeros(B + 1, device=q.device, dtype=torch.int32)
        cu[1:] = torch.cumsum(lengths, 0)

        sel = pad_mask.bool()  # (B,T)
        # pack tensors: permute to (B,T,H,D) then mask‑select → (ΣT, H, D)
        q_p = q.permute(0, 2, 1, 3)[sel]
        k_p = k.permute(0, 2, 1, 3)[sel]
        v_p = v.permute(0, 2, 1, 3)[sel]

        # call var‑len Sage kernel
        y_p = _sage_varlen(
            q_p,
            k_p,
            v_p,
            cu_seqlens_q=cu,
            cu_seqlens_k=cu,
            max_seqlen_q=max_len,
            max_seqlen_k=max_len,
            is_causal=False,
        )  # (ΣT, H, D)

        # scatter back to padded tensor
        y = torch.zeros_like(q)  # (B,H,T,D)
        y_perm = y.permute(0, 2, 1, 3)  # (B,T,H,D)
        y_perm[sel] = y_p
        return y_perm.permute(0, 2, 1, 3)  # (B,H,T,D)

    # ---------------------------------------------------------------------
    def forward(self, x: torch.Tensor, pad_mask: Optional[torch.Tensor] = None):
        """x: (B,T,C); pad_mask: (B,T) with True=keep, False=pad"""
        B, T, C = x.size()

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        # reshape to (B,H,T,D)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2).contiguous()
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2).contiguous()
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2).contiguous()

        # ---------------- choose kernel ----------------------------------
        if self.use_sparge:
            y = _sparge_attn(q, k, v, simthreshd1=0.5, cdfthreshd=0.97, is_causal=False)

        elif self.use_sage:
            all_equal = pad_mask is None or pad_mask.all()
            if all_equal:
                # fast path – no packing
                y = _sage_direct(q, k, v, is_causal=False)
            else:
                y = self._pack_and_sage(q, k, v, pad_mask)

        elif hasattr(F, "scaled_dot_product_attention"):
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                key_padding_mask=pad_mask,
                dropout_p=self.dropout,
                is_causal=False,
            )
        else:
            # fallback softmax path (no pad‑mask support)
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = F.softmax(att, dim=-1)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class FineBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = NonCausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)  # type: ignore – imported from outer scope

    def forward(self, x: torch.Tensor, pad_mask: Optional[torch.Tensor] = None):
        x = x + self.attn(self.ln_1(x), pad_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class FineGPT(GPT):  # GPT imported from outer scope
    """Same API as before, plus optional pad_mask."""

    def __init__(self, config):
        super().__init__(config)
        del self.lm_head  # remove causal head inherited from GPT
        self.config = config
        self.n_codes_total = config.n_codes_total

        self.transformer = nn.ModuleDict(
            dict(
                wtes=nn.ModuleList(
                    [
                        nn.Embedding(config.input_vocab_size, config.n_embd)
                        for _ in range(config.n_codes_total)
                    ]
                ),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([FineBlock(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )
        self.lm_heads = nn.ModuleList(
            [
                nn.Linear(config.n_embd, config.output_vocab_size, bias=False)
                for _ in range(config.n_codes_given, self.n_codes_total)
            ]
        )
        for i in range(self.n_codes_total - config.n_codes_given):
            self.transformer.wtes[i + 1].weight = self.lm_heads[i].weight

    def forward(self, pred_idx: int, idx: torch.Tensor, *, pad_mask: Optional[torch.Tensor] = None):
        device = idx.device
        b, t, codes = idx.size()
        assert t <= self.config.block_size
        assert codes == self.n_codes_total
        assert pred_idx > 0

        pos = torch.arange(0, t, device=device).unsqueeze(0)  # (1,t)

        # embed & sum across codebooks we’re conditioning on
        tok_embs = [wte(idx[:, :, i]).unsqueeze(-1) for i, wte in enumerate(self.transformer.wtes)]
        tok_emb = torch.cat(tok_embs, dim=-1)  # (B,T,C,n_codes)
        x = tok_emb[:, :, :, : pred_idx + 1].sum(dim=-1) + self.transformer.wpe(pos)
        x = self.transformer.drop(x)

        # transformer blocks
        for block in self.transformer.h:
            x = block(x, pad_mask)

        x = self.transformer.ln_f(x)
        logits = self.lm_heads[pred_idx - self.config.n_codes_given](x)
        return logits

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            for wte in self.transformer.wtes:
                n_params -= wte.weight.numel()
            n_params -= self.transformer.wpe.weight.numel()
        return n_params


@dataclass
class FineGPTConfig(GPTConfig):
    n_codes_total: int = 8
    n_codes_given: int = 1

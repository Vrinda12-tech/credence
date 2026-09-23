"""Four sequence models behind ONE interface, built from open-source libraries.

  lstm         torch.nn.LSTM                              gated RNN (Hochreiter & Schmidhuber 1997)
  transformer  transformers.GPT2Model                     causal softmax attention (Vaswani 2017)
  rwkv         transformers.RwkvModel  (RWKV-4)           linear attention, fixed per-channel decay
  mamba        mambapy.Mamba (pure PyTorch parallel scan) selective SSM, input-dependent Delta_t
  mamba_noconv ablation: Mamba with conv width 1 (no built-in delay taps)
  mamba_hf     transformers.MambaModel (slow reference)   same architecture, used to cross-check

All run on CPU (no CUDA kernels needed).  Interface:
    model(tokens: LongTensor (B,T)) -> logits (B,T,M), hidden (B,T,d)
Token x_t is a mixed-radix code of env.token_fields (abstract POMDP: (obs, action); grid: (position, reading, action)).
logits[:, t] parametrise the law of the next target given x_{0:t}.
`hidden` is the last-layer representation AFTER the final norm; belief probes read this.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

ARCHS = ["lstm", "transformer", "rwkv", "mamba"]
EXTRA_ARCHS = ["delay_mlp", "mamba_noconv"]   # classical null (literal delay embedding) and the conv-tap ablation
STRUCTURED_ARCHS = ["koopman", "koopman_pure", "neuralbayes"]   # controllable hidden state (structured.py); n_state is set from env
DELAY_WINDOW = 8


class DelayMLP(nn.Module):
    """LITERAL delay embedding: an MLP on the last W embedded tokens (zero-padded on the left).  Causal, no recurrence, no
    attention: the classical null model for the delay-embedding hypothesis (it sees exactly W lags and nothing else)."""

    def __init__(self, d: int, window: int = DELAY_WINDOW):
        super().__init__()
        self.window = window
        self.net = nn.Sequential(nn.Linear(window * d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))

    def forward(self, x):
        xp = F.pad(x, (0, 0, self.window - 1, 0))
        win = xp.unfold(1, self.window, 1).permute(0, 1, 3, 2).reshape(x.shape[0], x.shape[1], -1)
        return self.net(win)


class SeqModel(nn.Module):
    def __init__(self, arch: str, fields, n_out: int, d_model: int, n_layers: int = 2,
                 max_len: int = 128, d_state: int = 16, in_dim: int | None = None, n_state: int | None = None):
        super().__init__()
        self.arch, self.n_out, self.d, self.in_dim = arch, n_out, d_model, in_dim
        self.fields = tuple(fields) if fields is not None else ()
        if in_dim is not None:            # continuous inputs (Kalman-filter environments)
            self.inproj = nn.Linear(in_dim, d_model)
            self.embeds = nn.ModuleList()
        else:                             # token = mixed-radix code of the fields, e.g. (position, reading, action)
            self.embeds = nn.ModuleList([nn.Embedding(f, d_model) for f in self.fields])
        self.final_norm = nn.Identity()
        if arch == "lstm":
            self.core = nn.LSTM(d_model, d_model, num_layers=n_layers, batch_first=True)
            self.final_norm = nn.LayerNorm(d_model)
        elif arch == "transformer":
            from transformers import GPT2Config, GPT2Model
            cfg = GPT2Config(vocab_size=8, n_positions=max_len, n_embd=d_model, n_layer=n_layers,
                             n_head=4, resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,
                             bos_token_id=0, eos_token_id=0)
            self.core = GPT2Model(cfg)
            self.core.wte = None  # we feed inputs_embeds; drop unused table so it doesn't count as params
        elif arch == "rwkv":
            from transformers import RwkvConfig, RwkvModel
            cfg = RwkvConfig(vocab_size=8, context_length=max_len, hidden_size=d_model,
                             num_hidden_layers=n_layers, attention_hidden_size=d_model,
                             intermediate_size=4 * d_model, bos_token_id=0, eos_token_id=0)
            self.core = RwkvModel(cfg)
            self.core.embeddings = None
        elif arch in ("mamba", "mamba_noconv"):
            from mambapy.mamba import Mamba, MambaConfig
            # mamba_noconv: ablation that removes the width-4 causal conv (Mamba's built-in FIR delay taps)
            self.core = Mamba(MambaConfig(d_model=d_model, n_layers=n_layers, d_state=d_state,
                                          expand_factor=2, d_conv=1 if arch == "mamba_noconv" else 4, pscan=True))
            self.final_norm = nn.LayerNorm(d_model)
        elif arch == "delay_mlp":
            self.core = DelayMLP(d_model)
            self.final_norm = nn.LayerNorm(d_model)
        elif arch in ("koopman", "koopman_pure"):
            from .structured import KoopmanCell
            self.core = KoopmanCell(d_model, selective=(arch == "koopman"))
            self.final_norm = nn.LayerNorm(d_model)
        elif arch == "neuralbayes":
            from .structured import NeuralBayesCell
            self.core = NeuralBayesCell(d_model, n_state=n_state or d_model, d_out=d_model)
            self.final_norm = nn.LayerNorm(d_model)
        elif arch == "mamba_hf":
            from transformers import MambaConfig, MambaModel
            cfg = MambaConfig(vocab_size=8, hidden_size=d_model, num_hidden_layers=n_layers,
                              state_size=d_state, expand=2, bos_token_id=0, eos_token_id=0, pad_token_id=0)
            self.core = MambaModel(cfg)
            self.core.embeddings = None
        else:
            raise ValueError(arch)
        self.head = nn.Linear(d_model, n_out)

    def embed(self, tokens: torch.Tensor):
        if self.in_dim is not None:
            return self.inproj(tokens)
        x, idx = 0, tokens
        for emb, size in zip(reversed(self.embeds), reversed(self.fields)):
            x = x + emb(idx % size)
            idx = idx // size
        return x

    def forward(self, tokens: torch.Tensor):
        x = self.embed(tokens)
        a = self.arch
        if a == "lstm":
            h, _ = self.core(x)
        elif a in ("transformer", "rwkv", "mamba_hf"):
            h = self.core(inputs_embeds=x).last_hidden_state
        elif a in ("mamba", "mamba_noconv", "delay_mlp", "koopman", "koopman_pure", "neuralbayes"):
            h = self.core(x)
        h = self.final_norm(h)
        return self.head(h), h

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def core_parameters(self):
        """The recurrent/attention stack (RESeL's 'context encoder'): where Luo et al. 2024 (NeurIPS, arXiv:2405.15384)
        prove output perturbations are amplified across rollout length for any contractive hidden recurrence (K_h < 1),
        which GRU/LSTM (sigmoid gates), Mamba and RWKV (bounded per-channel decay) all satisfy."""
        return list(self.core.parameters())

    def head_parameters(self):
        """Everything NOT in the recurrent core: embeddings, input projection, final norm, output head.
        RESeL's finding: this part should NOT be slowed down with the core's learning rate."""
        core_ids = {id(p) for p in self.core.parameters()}
        return [p for p in self.parameters() if id(p) not in core_ids]


def build_model(arch: str, fields, n_out: int, d_model: int, n_layers: int = 2, max_len: int = 128):
    return SeqModel(arch, fields, n_out, d_model, n_layers, max_len)


def build_matched(arch: str, fields, n_out: int, target_params: int, n_layers: int = 2,
                  max_len: int = 128, in_dim: int | None = None) -> SeqModel:
    """Bisect d_model (multiple of 4; #params is monotone in d) to hit `target_params` closely."""
    lo, hi = 4, 512  # in units of 4
    cache = {}

    def make(k):
        if k not in cache:
            cache[k] = SeqModel(arch, fields, n_out, 4 * k, n_layers, max_len, in_dim=in_dim)
        return cache[k]

    while hi - lo > 1:
        mid = (lo + hi) // 2
        if make(mid).n_params() < target_params:
            lo = mid
        else:
            hi = mid
    a, b = make(lo), make(hi)
    return a if abs(a.n_params() - target_params) <= abs(b.n_params() - target_params) else b

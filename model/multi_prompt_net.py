import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat


def exists(val):
    return val is not None

class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim=None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x_q, x_kv=None, **kwargs):
        x_q = self.norm(x_q)

        if exists(x_kv):
            x_kv = self.norm_context(x_kv)
        else:
            x_kv = x_q

        return self.fn(x_q, x_kv, x_kv, **kwargs)

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)

class MultiCrossAttention(nn.Module):
    def __init__(self, latent_dim, kv_dim, cross_heads=4, seq_dropout_prob=0.0):
        super().__init__()
        self.cross_attend_blocks = nn.ModuleList([
            PreNorm(
                latent_dim,
                nn.MultiheadAttention(
                    latent_dim,
                    num_heads=cross_heads,
                    kdim=kv_dim,
                    vdim=kv_dim,
                    dropout=seq_dropout_prob,
                    batch_first=True,
                ),
                context_dim=kv_dim,
            ),
            FeedForward(latent_dim),
        ])

    def forward(self, data, soft_prompt, mask=None):
        # data: [B, T, C], soft_prompt: [G, L, C]
            # B: batch size (here is 1)
            # G: number of prompts
            # L: prompt_len * prompt_depth (Flattened length of prompts)
            # C: prompt dimension (channel, feature dim -> 512)
            # T: number of condition token (#classes)
        g = soft_prompt.shape[0]
        if data.shape[0] == 1 and g > 1:
            data = repeat(data, "1 t c -> g t c", g=g)
        elif data.shape[0] != g:
            raise ValueError(
                f"Batch mismatch in MultiCrossAttention: data batch={data.shape[0]}, prompts={g}"
            )

        x = soft_prompt
        cross_attn, cross_ff = self.cross_attend_blocks
        x, _ = cross_attn(x, data, key_padding_mask=mask)
        x = cross_ff(x) + x
        return x


class MultiSelfAttention(nn.Module):
    def __init__(self, depth, latent_dim, latent_heads=4):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            latent_dim,
                            nn.MultiheadAttention(
                                latent_dim, num_heads=latent_heads, batch_first=True
                            ),
                        ),
                        FeedForward(latent_dim),
                    ]
                )
            )

    def forward(self, x, mask=None):
        for self_attn, self_ff in self.layers:
            x = self_attn(x, key_padding_mask=mask)[0] + x
            x = self_ff(x) + x
        return x


class MultiPromptTranslator(nn.Module):
    def __init__(
        self,
        prompt_len,
        prompt_depth,
        num_prompts,
        prompt_dim=512,
        depth=4,
        self_heads=4,
        cross_heads=4,
        textemb_dim=512,
    ):
        super().__init__()
        self.prompt_len = prompt_len
        self.prompt_depth = prompt_depth
        self.num_prompts = num_prompts
        self.depth = depth

        soft_prompt = torch.empty(num_prompts, prompt_len * prompt_depth, prompt_dim)
        nn.init.normal_(soft_prompt, std=0.02)
        self.soft_prompt = nn.Parameter(soft_prompt)

        self.encoder = MultiCrossAttention(
            latent_dim=prompt_dim,
            kv_dim=textemb_dim,
            cross_heads=cross_heads,
        )
        if depth > 0:
            self.transformer = MultiSelfAttention(
                depth=depth, latent_dim=prompt_dim, latent_heads=self_heads
            )

    def forward(self, text_emb):
        # text_emb: [B, T, C], usually B=1 (a client's class name set embedding)
        prompt = self.encoder(text_emb, self.soft_prompt)
        if self.depth > 0:
            prompt = self.transformer(prompt)
        prompt = prompt.reshape(
            self.num_prompts, self.prompt_depth, self.prompt_len, -1
        )
        return prompt, prompt

import torch
import torch.nn as nn
import torch.nn.functional as F

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from model.multi_prompt_net import MultiPromptTranslator

_tokenizer = _Tokenizer()


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ImageEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.conv1 = clip_model.conv1
        self.class_embedding = clip_model.class_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.ln_pre = clip_model.ln_pre
        self.transformer = clip_model.transformer
        self.ln_post = clip_model.ln_post
        self.proj = clip_model.proj

    def forward(self, x, vis_ctx=[]):
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat(
            [
                self.class_embedding.to(x.dtype)
                + torch.zeros(
                    x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
                ),
                x,
            ],
            dim=1,
        )
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)
        x = self.transformer(x, vis_ctx, False)
        x = x.permute(1, 0, 2)
        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj
        return x


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, text_ctx):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x, text_ctx, True)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


class MultiPromptLearner(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        n_ctx, ctx_depth = cfg.MODEL.N_CTX, cfg.MODEL.D_CTX
        num_prompts = cfg.MODEL.NUM_PROMPTS
        self.meta_net = MultiPromptTranslator(
            n_ctx, ctx_depth, num_prompts=num_prompts, depth=cfg.MODEL.DEPTH
        )
        self.meta_net.half()

    def forward(self, context_emb):
        # context_emb: [num_classes, 512]
        text_ctx_pool, vis_ctx_pool = self.meta_net(context_emb.unsqueeze(0))
        # -> [G, D_CTX, N_CTX, 512]
        return text_ctx_pool, vis_ctx_pool


class ClientGatingNet(nn.Module):
    def __init__(self, in_dim, num_prompts, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            QuickGELU(),
            nn.Linear(hidden_dim, num_prompts),
        )

    def forward(self, client_emb):
        return self.net(client_emb)


class FedMoPG(nn.Module):
    def __init__(self, cfg, clip_model, device="cuda"):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.dtype = clip_model.dtype
        self.clip_model_ = clip_model

        self.set_prompt_prefix()
        self.prompt_learner = MultiPromptLearner(cfg)
        self.gating_net = ClientGatingNet(
            in_dim=512,
            num_prompts=cfg.MODEL.NUM_PROMPTS,
            hidden_dim=cfg.MODEL.GATE_HIDDEN_DIM,
        )

        self.image_encoder = ImageEncoder(clip_model.visual)
        self.text_encoder = TextEncoder(clip_model)
        self.token_embedding = clip_model.token_embedding
        self.logit_scale = clip_model.logit_scale

        self.num_prompts = cfg.MODEL.NUM_PROMPTS
        self.top_k = max(1, min(cfg.MODEL.TOP_K, self.num_prompts))
        self.gate_reg_weight = cfg.MODEL.GATE_REG_WEIGHT
        self.diversity_reg_weight = cfg.MODEL.DIVERSITY_REG_WEIGHT
        self.debug_print_every = 1
        self._train_step = 0

    def set_prompt_prefix(self):
        self.n_ctx = self.cfg.MODEL.N_CTX
        self.prompt_prefix = " ".join(["X"] * self.n_ctx)
        print(f'Initial context: "{self.prompt_prefix}"')
        print(f"Number of context words (tokens): {self.n_ctx}")

    def get_tokenized_classnames(self, classnames):
        prompts = [self.prompt_prefix + " " + name + "." for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = self.token_embedding(tokenized_prompts.to(self.device)).type(self.dtype)
        return embedding, tokenized_prompts

    def _encode_text_with_prompt(self, classnames, text_ctx):
        prompt_vectors, tokenized_prompts = self.get_tokenized_classnames(classnames)

        prompt_vectors = torch.cat(
            [
                prompt_vectors[:, :1],
                text_ctx[0].unsqueeze(0).expand(prompt_vectors.shape[0], -1, -1),
                prompt_vectors[:, 1 + text_ctx.shape[1] :],
            ],
            dim=1,
        )

        if len(text_ctx) > 1:
            deep_ctx = text_ctx[1:]
        else:
            deep_ctx = []
        return self.text_encoder(prompt_vectors, tokenized_prompts, deep_ctx)

    def _gate_regularizer(self, gate_probs):
        # KL(p || uniform): minimize to avoid prompt-wise over-specialization
        eps = 1e-7
        uniform = 1.0 / gate_probs.shape[-1]
        kl = gate_probs * (
            torch.log(gate_probs + eps)
            - torch.log(torch.tensor(uniform, device=gate_probs.device))
        )
        return kl.sum()

    def _prompt_diversity_regularizer(self, text_ctx_pool):
        """
        text_ctx_pool: [G, D_CTX, N_CTX, C]
        G: number of prompts (ex. 4)
        D_CTX: number of layers (ex. 1) = number of transformer depth 
        N_CTX: number of context tokens (ex. 4)
        C: prompt embedding dimension (ex. 512)
        Penalize pairwise cosine similarity among prompt experts.
        """
        g = text_ctx_pool.shape[0]
        if g <= 1:
            return torch.tensor(0.0, device=text_ctx_pool.device, dtype=text_ctx_pool.dtype)

        prompt_flat = text_ctx_pool.reshape(g, -1).float()
        prompt_flat = F.normalize(prompt_flat, dim=-1)
        sim = prompt_flat @ prompt_flat.t()  # [G, G]
        eye = torch.eye(g, device=sim.device, dtype=torch.bool)
        off_diag = sim[~eye]
        return (off_diag ** 2).mean()

    def _prompt_similarity_matrix(self, text_ctx_pool):
        g = text_ctx_pool.shape[0]
        prompt_flat = text_ctx_pool.reshape(g, -1).float()
        prompt_flat = F.normalize(prompt_flat, dim=-1)
        return prompt_flat @ prompt_flat.t()

    def forward(self, image, classnames, dataname):
        del dataname
        classnames = [name.replace("_", " ") for name in classnames]
        if self.training:
            self._train_step += 1

        # Client-specific condition embedding from raw class names.
        prompts_ = torch.cat([clip.tokenize(p) for p in classnames]).to(self.device)
        with torch.no_grad():
            context_emb = self.clip_model_.encode_text(prompts_)
            context_emb = context_emb / context_emb.norm(dim=-1, keepdim=True)

        # [G, D_CTX, N_CTX, 512]
        text_ctx_pool, _ = self.prompt_learner(context_emb)

        # Local gating chooses top-k prompts for this client.
        client_emb = context_emb.mean(dim=0, keepdim=True).float()
        gate_logits = self.gating_net(client_emb).squeeze(0)
        gate_probs = F.softmax(gate_logits, dim=-1)
        topk_probs, topk_indices = torch.topk(gate_probs, k=self.top_k, dim=-1)
        topk_weights = topk_probs / topk_probs.sum()

        if self.training and self.debug_print_every > 0 and (self._train_step % self.debug_print_every == 0):
            with torch.no_grad():
                sim = self._prompt_similarity_matrix(text_ctx_pool)
                g = sim.shape[0]
                eye = torch.eye(g, device=sim.device, dtype=torch.bool)
                off_diag = sim[~eye]
                print(
                    f"[FedMoPG Debug][step={self._train_step}] "
                    f"gate_probs={gate_probs.detach().cpu().tolist()} "
                    f"topk_indices={topk_indices.detach().cpu().tolist()} "
                    f"topk_weights={topk_weights.detach().cpu().tolist()} "
                    f"offdiag_min={off_diag.min().item():.4f} "
                    f"offdiag_max={off_diag.max().item():.4f}"
                )
                print(f"[FedMoPG Debug][step={self._train_step}] prompt_sim_matrix={sim.detach().cpu().tolist()}")

        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        logits = 0.0
        for w, idx in zip(topk_weights, topk_indices):
            text_ctx = text_ctx_pool[idx].to(self.dtype)
            text_features = self._encode_text_with_prompt(classnames, text_ctx)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            logits = logits + w * (self.logit_scale.exp() * image_features @ text_features.t())

        gate_reg = self._gate_regularizer(gate_probs)
        diversity_reg = self._prompt_diversity_regularizer(text_ctx_pool)
        reg = self.gate_reg_weight * gate_reg + self.diversity_reg_weight * diversity_reg
        reg_terms = {
            "gate_reg": gate_reg,
            "diversity_reg": diversity_reg,
        }
        return logits, reg, reg_terms

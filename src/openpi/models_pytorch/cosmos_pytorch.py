"""pi05 with Cosmos-Reason2 (Qwen3-VL) as the VLM, in PyTorch.

Layout and attention (same idea as pi0/pi05, see pi0_pytorch.py):

    prefix = [<vision_start> img_0 <vision_end>] ... [<vision_start> img_2 <vision_end>] prompt_tokens
    suffix = action tokens (noisy actions, timestep injected through adaRMSNorm)

At every decoder layer the suffix queries attend to the prefix keys/values of the same layer. Prefix tokens never attend
to the suffix, so the prefix can be run first on its own (collecting per-layer K/V) and the action expert second. That is
exactly equivalent to the joint per-layer forward in gemma_pytorch.py, and it lets a frozen VLM run under no_grad.

Requires transformers>=4.57 (Qwen3-VL). It does not need the `transformers_replace` patches used by PI0Pytorch.
"""

import logging
import math

import safetensors.torch
import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F  # noqa: N812

from openpi.models import cosmos_config
from openpi.models_pytorch.pi0_pytorch import create_sinusoidal_pos_embedding
from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
from openpi.models_pytorch.pi0_pytorch import sample_beta
import openpi.models_pytorch.preprocessing_pytorch as _preprocessing

logger = logging.getLogger("openpi")


def _rotate_half(x: Tensor) -> Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
    """q, k: [B, H, L, D]; cos, sin: [B, L, D]."""
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)


class LoRALinear(nn.Module):
    """Frozen nn.Linear plus a trainable low-rank update. B is zero-initialized, so it starts as the base layer."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)  # noqa: FBT003
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features, dtype=torch.float32))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self.scaling = alpha / rank

    @property
    def weight(self) -> Tensor:
        return self.base.weight

    def forward(self, x: Tensor) -> Tensor:
        lora = F.linear(F.linear(x, self.lora_a.to(x.dtype)), self.lora_b.to(x.dtype))
        return self.base(x) + lora * self.scaling


class RMSNorm(nn.Module):
    """Qwen-style RMSNorm (weight initialized to ones, applied as `w * x`)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(dtype)


class AdaRMSNorm(nn.Module):
    """RMSNorm modulated by a conditioning vector (the flow-matching timestep): returns (normed, gate).

    Same semantics as the adaRMS in openpi's patched Gemma: scale/shift/gate come from a zero-initialized dense layer.
    """

    def __init__(self, dim: int, cond_dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.dense = nn.Linear(cond_dim, dim * 3, bias=True)
        nn.init.zeros_(self.dense.weight)

    def forward(self, x: Tensor, cond: Tensor) -> tuple[Tensor, Tensor]:
        dtype = x.dtype
        xf = x.float()
        normed = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        scale, shift, gate = self.dense(cond)[:, None, :].chunk(3, dim=-1)
        normed = normed * (1.0 + scale.float()) + shift.float()
        return normed.to(dtype), gate.to(dtype)


class ActionExpertLayer(nn.Module):
    def __init__(self, width: int, mlp_dim: int, num_heads: int, num_kv_heads: int, head_dim: int, eps: float):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.input_layernorm = AdaRMSNorm(width, width, eps)
        self.q_proj = nn.Linear(width, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(width, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(width, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, width, bias=False)
        self.q_norm = RMSNorm(head_dim, eps)
        self.k_norm = RMSNorm(head_dim, eps)
        self.post_attention_layernorm = AdaRMSNorm(width, width, eps)
        self.gate_proj = nn.Linear(width, mlp_dim, bias=False)
        self.up_proj = nn.Linear(width, mlp_dim, bias=False)
        self.down_proj = nn.Linear(mlp_dim, width, bias=False)

    def forward(
        self,
        h: Tensor,
        cond: Tensor,
        cos: Tensor,
        sin: Tensor,
        prefix_k: Tensor,
        prefix_v: Tensor,
        attn_mask: Tensor,
    ) -> Tensor:
        bsize, seq_len, _ = h.shape
        x, gate = self.input_layernorm(h, cond)
        q = self.q_norm(self.q_proj(x).view(bsize, seq_len, -1, self.head_dim)).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(bsize, seq_len, -1, self.head_dim)).transpose(1, 2)
        v = self.v_proj(x).view(bsize, seq_len, -1, self.head_dim).transpose(1, 2)
        q, k = _apply_rope(q, k, cos, sin)
        k = torch.cat([prefix_k.to(k.dtype), k], dim=2)
        v = torch.cat([prefix_v.to(v.dtype), v], dim=2)
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, enable_gqa=True)
        attn = attn.transpose(1, 2).reshape(bsize, seq_len, -1)
        h = h + self.o_proj(attn) * gate

        x, gate = self.post_attention_layernorm(h, cond)
        x = self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        return h + x * gate


class ActionExpert(nn.Module):
    def __init__(self, depth: int, width: int, mlp_dim: int, num_heads: int, num_kv_heads: int, head_dim: int, eps: float):
        super().__init__()
        self.layers = nn.ModuleList(
            [ActionExpertLayer(width, mlp_dim, num_heads, num_kv_heads, head_dim, eps) for _ in range(depth)]
        )
        self.norm = AdaRMSNorm(width, width, eps)
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) and not name.endswith("dense"):
                nn.init.normal_(module.weight, std=0.02)


class CosmosPi05Pytorch(nn.Module):
    def __init__(self, config: cosmos_config.CosmosPi05Config, *, load_vlm_weights: bool = True):
        super().__init__()
        from transformers import AutoConfig
        from transformers import Qwen3VLForConditionalGeneration

        self.config = config
        if load_vlm_weights:
            full = Qwen3VLForConditionalGeneration.from_pretrained(config.vlm_path, dtype=torch.bfloat16)
        else:
            full = Qwen3VLForConditionalGeneration._from_config(  # noqa: SLF001
                AutoConfig.from_pretrained(config.vlm_path), dtype=torch.bfloat16
            )
        # Keep only the vision tower + text decoder; the LM head (~0.6B params) is never used.
        self.vlm = full.model
        del full
        vlm_cfg = self.vlm.config
        text_cfg = vlm_cfg.text_config
        vis_cfg = vlm_cfg.vision_config

        self.vision_start_id = vlm_cfg.vision_start_token_id
        self.vision_end_id = vlm_cfg.vision_end_token_id
        self.patch_size = vis_cfg.patch_size
        self.temporal_patch_size = vis_cfg.temporal_patch_size
        self.merge_size = vis_cfg.spatial_merge_size
        self.num_llm_layers = text_cfg.num_hidden_layers
        self.head_dim = getattr(text_cfg, "head_dim", text_cfg.hidden_size // text_cfg.num_attention_heads)

        grid = config.image_resolution // self.patch_size
        self.llm_grid = grid // self.merge_size  # merged tokens per side
        self.tokens_per_image = self.llm_grid**2

        width = config.action_expert_width
        self.action_expert = ActionExpert(
            depth=text_cfg.num_hidden_layers,
            width=width,
            mlp_dim=config.action_expert_mlp_dim,
            num_heads=text_cfg.num_attention_heads,
            num_kv_heads=text_cfg.num_key_value_heads,
            head_dim=self.head_dim,
            eps=text_cfg.rms_norm_eps,
        )
        self.action_in_proj = nn.Linear(config.action_dim, width)
        self.action_out_proj = nn.Linear(width, config.action_dim)
        self.time_mlp_in = nn.Linear(width, width)
        self.time_mlp_out = nn.Linear(width, width)

        self._setup_trainable()
        self.gradient_checkpointing_enabled = False
        torch.set_float32_matmul_precision("high")

    # ------------------------------------------------------------------------------------------------------------------
    # Parameter freezing / LoRA / checkpoint IO
    # ------------------------------------------------------------------------------------------------------------------

    def _setup_trainable(self):
        cfg = self.config
        self.vlm.requires_grad_(False)  # noqa: FBT003
        if not cfg.freeze_vision:
            self.vlm.visual.requires_grad_(True)  # noqa: FBT003
        elif cfg.train_vision_merger:
            self.vlm.visual.merger.requires_grad_(True)  # noqa: FBT003
            self.vlm.visual.deepstack_merger_list.requires_grad_(True)  # noqa: FBT003
        layers = self.vlm.language_model.layers
        num_layers = self.num_llm_layers
        n_full = min(cfg.train_llm_last_n_layers, num_layers)
        if n_full > 0:
            for layer in layers[num_layers - n_full :]:
                layer.requires_grad_(True)  # noqa: FBT003
            # The expert only consumes the last layer's K/V, so the rest of that layer cannot affect the loss.
            last = layers[num_layers - 1]
            for module in (last.self_attn.q_proj, last.self_attn.q_norm, last.self_attn.o_proj,
                           last.post_attention_layernorm, last.mlp):
                module.requires_grad_(False)  # noqa: FBT003
        if cfg.llm_lora_rank > 0:
            targets = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
            for layer in layers[: num_layers - n_full]:  # fully trained layers need no LoRA
                for parent in (layer.self_attn, layer.mlp):
                    for name in targets:
                        if isinstance(getattr(parent, name, None), nn.Linear):
                            setattr(parent, name, LoRALinear(getattr(parent, name), cfg.llm_lora_rank, cfg.llm_lora_alpha))

        # Trainable VLM params are kept in float32 (bf16 weights cannot absorb small Adam updates); frozen ones stay bf16.
        # The action expert and projections are float32; the forward runs under bf16 autocast.
        for p in self.vlm.parameters():
            if p.requires_grad:
                p.data = p.data.float()

        self.vlm_requires_grad = any(p.requires_grad for p in self.vlm.parameters())
        # Vision tower / merger / LoRA need grads through the whole prefix; last-N-layers only from layer L-N up.
        self.vision_requires_grad = any(p.requires_grad for p in self.vlm.visual.parameters())
        if self.vision_requires_grad or cfg.llm_lora_rank > 0:
            self.first_grad_layer = 0
        elif n_full > 0:
            self.first_grad_layer = num_layers - n_full
        else:
            self.first_grad_layer = num_layers
        n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.parameters())
        n_expert = sum(p.numel() for p in self.action_expert.parameters())
        n_vlm = sum(p.numel() for p in self.vlm.parameters() if p.requires_grad)
        logger.info(
            f"CosmosPi05: trainable {n_train / 1e6:.1f}M / total {n_total / 1e6:.1f}M params "
            f"(action expert {n_expert / 1e6:.1f}M, VLM trainable {n_vlm / 1e6:.1f}M, "
            f"fully trained LLM layers: {n_full}, grads from LLM layer {self.first_grad_layer})"
        )

    def param_groups(self) -> list[dict]:
        """AdamW param groups: trainable VLM params get `lr_scale=vlm_lr_multiplier`, the rest `lr_scale=1`.

        The trainer multiplies the scheduled LR by each group's `lr_scale`. Group 0 is the action expert group.
        """
        vlm_ids = {id(p) for p in self.vlm.parameters()}
        vlm_params, other_params = [], []
        for p in self.parameters():
            if p.requires_grad:
                (vlm_params if id(p) in vlm_ids else other_params).append(p)
        groups = [{"params": other_params, "lr_scale": 1.0}]
        if vlm_params:
            groups.append({"params": vlm_params, "lr_scale": self.config.vlm_lr_multiplier})
        return groups

    def trainable_state_dict(self) -> dict[str, Tensor]:
        """Only the trainable parameters. The frozen VLM is reloaded from `vlm_path`, so it is not saved."""
        return {n: p.detach().contiguous() for n, p in self.named_parameters() if p.requires_grad}

    def load_trainable_weights(self, path: str, device: str | torch.device = "cpu"):
        state = safetensors.torch.load_file(str(path), device=str(device))
        missing, unexpected = self.load_state_dict(state, strict=False)
        expected = {n for n, p in self.named_parameters() if p.requires_grad}
        missing_trainable = sorted(set(missing) & expected)
        # Trainable VLM params absent from the checkpoint keep their pretrained values (LoRA B starts at zero). This is
        # what happens when a frozen-VLM checkpoint warm-starts a config that trains part of the VLM.
        missing_vlm = [n for n in missing_trainable if n.startswith("vlm.")]
        missing_other = [n for n in missing_trainable if not n.startswith("vlm.")]
        if missing_other or unexpected:
            raise ValueError(
                f"Checkpoint {path} does not match the model: missing trainable {missing_other[:10]}, "
                f"unexpected {unexpected[:10]}"
            )
        if missing_vlm:
            logger.warning(
                f"{len(missing_vlm)} trainable VLM tensors not in {path}; keeping their pretrained values "
                f"(e.g. {missing_vlm[:3]}). Expected when warm-starting from a frozen-VLM checkpoint."
            )
        logger.info(f"Loaded {len(state)} trained tensors from {path}")

    def gradient_checkpointing_enable(self):
        self.gradient_checkpointing_enabled = True

    def gradient_checkpointing_disable(self):
        self.gradient_checkpointing_enabled = False

    def _ckpt(self, fn, *args):
        if self.gradient_checkpointing_enabled and self.training and torch.is_grad_enabled():
            return torch.utils.checkpoint.checkpoint(fn, *args, use_reentrant=False, preserve_rng_state=False)
        return fn(*args)

    # ------------------------------------------------------------------------------------------------------------------
    # Prefix: images + prompt through Qwen3-VL, collecting per-layer K/V
    # ------------------------------------------------------------------------------------------------------------------

    def _patchify(self, images: Tensor) -> tuple[Tensor, Tensor]:
        """[N, 3, H, W] in [-1, 1] -> Qwen3-VL flattened patches + grid_thw (same as Qwen2VLImageProcessor).

        Qwen3-VL normalizes with mean=std=0.5, i.e. (x/255 - 0.5) / 0.5, which is exactly openpi's [-1, 1] range.
        """
        n, c, h, w = images.shape
        p, m, t = self.patch_size, self.merge_size, self.temporal_patch_size
        gh, gw = h // p, w // p
        x = images[:, None].expand(n, t, c, h, w)  # a still image is repeated along the temporal patch axis
        x = x.reshape(n, 1, t, c, gh // m, m, p, gw // m, m, p)
        x = x.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
        x = x.reshape(n * gh * gw, c * t * p * p)
        grid_thw = torch.tensor([[1, gh, gw]], device=images.device, dtype=torch.long).expand(n, 3)
        return x, grid_thw

    def _to_channels_first(self, img: Tensor) -> Tensor:
        return img if img.shape[1] == 3 else img.permute(0, 3, 1, 2)

    def embed_prefix(self, images, img_masks, lang_tokens, lang_masks):
        """Returns embeddings, pad masks, attention (ar) masks, 3D M-RoPE positions, visual index, deepstack features."""
        bsize = lang_tokens.shape[0]
        device = lang_tokens.device
        n_img = len(images)
        tpi = self.tokens_per_image
        embed_tokens = self.vlm.language_model.embed_tokens

        pixel_values = torch.cat(
            [self._to_channels_first(img) for img in images], dim=0
        )  # [n_img * B, 3, H, W], camera-major
        size = self.config.image_resolution
        if pixel_values.shape[-2:] != (size, size):
            pixel_values = F.interpolate(pixel_values.float(), size=(size, size), mode="bilinear", align_corners=False)
        patches, grid_thw = self._patchify(pixel_values)
        patches = patches.to(self.vlm.visual.patch_embed.proj.weight.dtype)
        img_emb, deepstack = self.vlm.visual(patches, grid_thw=grid_thw)

        def per_sample(x):  # [n_img * B * tpi, D] -> [B, n_img * tpi, D]
            return x.view(n_img, bsize, tpi, -1).transpose(0, 1).reshape(bsize, n_img * tpi, -1)

        img_emb = per_sample(img_emb)
        deepstack = [per_sample(d) for d in deepstack]

        special = embed_tokens(torch.tensor([self.vision_start_id, self.vision_end_id], device=device))
        vs_emb = special[0].expand(bsize, 1, -1)
        ve_emb = special[1].expand(bsize, 1, -1)

        embs, pad_masks, incs, rows, cols, visual_index = [], [], [], [], [], []
        grid = self.llm_grid
        r = torch.arange(grid, device=device).repeat_interleave(grid)
        c = torch.arange(grid, device=device).repeat(grid)
        img_inc = torch.zeros(tpi, dtype=torch.long, device=device)
        img_inc[-1] = grid  # the token after an image block starts at +max(h, w), as in Qwen3VLModel.get_rope_index
        zero1 = torch.zeros(1, dtype=torch.long, device=device)
        one1 = torch.ones(1, dtype=torch.long, device=device)
        pos = 0
        for i in range(n_img):
            mask = img_masks[i][:, None]
            embs += [vs_emb, img_emb[:, i * tpi : (i + 1) * tpi], ve_emb]
            pad_masks += [mask, mask.expand(bsize, tpi), mask]
            incs += [one1, img_inc, one1]
            rows += [zero1, r, zero1]
            cols += [zero1, c, zero1]
            visual_index.append(torch.arange(pos + 1, pos + 1 + tpi, device=device))
            pos += tpi + 2

        lang_emb = embed_tokens(lang_tokens)
        n_lang = lang_tokens.shape[1]
        embs.append(lang_emb)
        pad_masks.append(lang_masks)
        incs.append(torch.ones(n_lang, dtype=torch.long, device=device))
        rows.append(torch.zeros(n_lang, dtype=torch.long, device=device))
        cols.append(torch.zeros(n_lang, dtype=torch.long, device=device))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        inc = torch.cat(incs)[None, :] * pad_masks.long()  # padding does not advance the position
        base = torch.cumsum(inc, dim=1) - inc  # exclusive cumsum
        rows = torch.cat(rows)[None, :]
        cols = torch.cat(cols)[None, :]
        positions = torch.stack([base, base + rows, base + cols], dim=0)  # [3, B, L]
        next_pos = inc.sum(dim=1)  # first position after the prefix, per sample

        seq_len = embs.shape[1]
        if self.config.causal_prefix:
            att_masks = torch.ones(bsize, seq_len, dtype=torch.bool, device=device)
        else:
            att_masks = torch.zeros(bsize, seq_len, dtype=torch.bool, device=device)
        return embs, pad_masks, att_masks, positions, next_pos, torch.cat(visual_index), deepstack

    def _llm_layer(self, layer_idx, h, cos, sin, attn_mask, visual_index, deepstack_feat):
        layer = self.vlm.language_model.layers[layer_idx]
        attn = layer.self_attn
        bsize, seq_len, _ = h.shape
        x = layer.input_layernorm(h)
        q = attn.q_norm(attn.q_proj(x).view(bsize, seq_len, -1, self.head_dim)).transpose(1, 2)
        k = attn.k_norm(attn.k_proj(x).view(bsize, seq_len, -1, self.head_dim)).transpose(1, 2)
        v = attn.v_proj(x).view(bsize, seq_len, -1, self.head_dim).transpose(1, 2)
        q, k = _apply_rope(q, k, cos, sin)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, enable_gqa=True)
        h = h + attn.o_proj(out.transpose(1, 2).reshape(bsize, seq_len, -1))
        h = h + layer.mlp(layer.post_attention_layernorm(h))
        if deepstack_feat is not None:
            h = h.index_add(1, visual_index, deepstack_feat.to(h.dtype))
        return h, k, v

    def compute_prefix_kv(self, images, img_masks, lang_tokens, lang_masks):
        """Runs the VLM over the prefix. Returns per-layer (K, V), prefix pad mask and the next position id."""
        def grad_ctx(needs_grad: bool):  # noqa: FBT001
            return torch.enable_grad() if (needs_grad and self.training) else torch.no_grad()

        # Everything below `first_grad_layer` is frozen and runs under no_grad, so it stores no activations and the
        # backward pass stops there.
        with grad_ctx(self.first_grad_layer == 0):
            embs, pad_masks, att_masks, positions, next_pos, visual_index, deepstack = self.embed_prefix(
                images, img_masks, lang_tokens, lang_masks
            )
        att_2d = make_att_2d_masks(pad_masks, att_masks)
        # Padded query rows would otherwise attend to nothing (NaN in SDPA); let them see themselves.
        att_2d = att_2d | torch.eye(att_2d.shape[-1], dtype=torch.bool, device=att_2d.device)[None]
        attn_mask = att_2d[:, None]
        cos, sin = self.vlm.language_model.rotary_emb(embs, positions)

        h = embs
        kv = []
        for i in range(self.num_llm_layers):
            ds = deepstack[i] if i < len(deepstack) else None
            with grad_ctx(i >= self.first_grad_layer):
                h, k, v = self._ckpt(self._llm_layer, i, h, cos, sin, attn_mask, visual_index, ds)
            kv.append((k, v))
        return kv, pad_masks, next_pos

    # ------------------------------------------------------------------------------------------------------------------
    # Suffix: action expert
    # ------------------------------------------------------------------------------------------------------------------

    def embed_suffix(self, noisy_actions, timestep):
        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0, device=timestep.device
        ).to(torch.float32)
        cond = F.silu(self.time_mlp_out(F.silu(self.time_mlp_in(time_emb))))
        return self.action_in_proj(noisy_actions), cond

    def run_action_expert(self, prefix_kv, prefix_pad_masks, next_pos, noisy_actions, timestep):
        tokens, cond = self.embed_suffix(noisy_actions, timestep)
        bsize, suffix_len, _ = tokens.shape
        prefix_len = prefix_pad_masks.shape[1]

        # Suffix attends to all valid prefix tokens and bidirectionally within the action chunk.
        prefix_part = prefix_pad_masks[:, None, :].expand(bsize, suffix_len, prefix_len)
        suffix_part = torch.ones(bsize, suffix_len, suffix_len, dtype=torch.bool, device=tokens.device)
        attn_mask = torch.cat([prefix_part, suffix_part], dim=2)[:, None]

        pos = next_pos[:, None] + torch.arange(suffix_len, device=tokens.device)[None, :]
        cos, sin = self.vlm.language_model.rotary_emb(tokens, pos[None].expand(3, -1, -1))

        h = tokens
        for i, layer in enumerate(self.action_expert.layers):
            k, v = prefix_kv[i]
            # Not checkpointed: the suffix is only action_horizon tokens, so its activations are small.
            h = layer(h, cond, cos, sin, k, v, attn_mask)
        h, _ = self.action_expert.norm(h, cond)
        return self.action_out_proj(h.float())

    # ------------------------------------------------------------------------------------------------------------------
    # Training / inference entry points (same signatures as PI0Pytorch)
    # ------------------------------------------------------------------------------------------------------------------

    def _preprocess_observation(self, observation, *, train: bool):
        res = (self.config.image_resolution, self.config.image_resolution)
        observation = _preprocessing.preprocess_observation_pytorch(observation, train=train, image_resolution=res)
        return (
            list(observation.images.values()),
            list(observation.image_masks.values()),
            observation.tokenized_prompt,
            observation.tokenized_prompt_mask,
        )

    def _autocast(self, device):
        # Required, not just an optimisation: trainable params are fp32 while the frozen VLM is bf16.
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type in ("cuda", "cpu"))

    def forward(self, observation, actions, noise=None, time=None) -> Tensor:
        # Augment only in train mode, so a validation loss (model.eval()) sees clean images.
        images, img_masks, lang_tokens, lang_masks = self._preprocess_observation(observation, train=self.training)
        if noise is None:
            noise = torch.randn_like(actions, dtype=torch.float32)
        if time is None:
            time = sample_beta(1.5, 1.0, actions.shape[0], actions.device) * 0.999 + 0.001
        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        with self._autocast(actions.device):
            prefix_kv, prefix_pad_masks, next_pos = self.compute_prefix_kv(images, img_masks, lang_tokens, lang_masks)
            v_t = self.run_action_expert(prefix_kv, prefix_pad_masks, next_pos, x_t, time)
        return F.mse_loss(u_t, v_t.float(), reduction="none")

    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, num_steps=10) -> Tensor:
        bsize = observation.state.shape[0]
        if noise is None:
            noise = torch.randn(bsize, self.config.action_horizon, self.config.action_dim, device=device)
        images, img_masks, lang_tokens, lang_masks = self._preprocess_observation(observation, train=False)

        with self._autocast(torch.device(device) if isinstance(device, str) else device):
            prefix_kv, prefix_pad_masks, next_pos = self.compute_prefix_kv(images, img_masks, lang_tokens, lang_masks)
            dt = -1.0 / num_steps
            x_t = noise.float()
            time = 1.0
            while time >= -dt / 2:
                t = torch.full((bsize,), time, dtype=torch.float32, device=x_t.device)
                v_t = self.run_action_expert(prefix_kv, prefix_pad_masks, next_pos, x_t, t)
                x_t = x_t + dt * v_t.float()
                time += dt
        return x_t

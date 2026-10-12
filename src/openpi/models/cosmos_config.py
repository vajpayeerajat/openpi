"""Config for pi05 with Cosmos-Reason2 (Qwen3-VL architecture) as the VLM backbone.

This model is PyTorch-only (train with `scripts/train_pytorch.py`). It keeps the pi05 recipe -- discrete state in the
prompt, flow-matching action expert conditioned on the timestep through adaRMSNorm, action expert attending to the VLM's
keys/values at every layer -- but swaps PaliGemma for Cosmos-Reason2 and builds a new action expert whose attention
geometry (depth, heads, kv heads, head_dim) matches the Qwen3-VL text decoder.
"""

import dataclasses
import logging
from typing import TYPE_CHECKING

from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.shared import array_typing as at

if TYPE_CHECKING:
    from openpi.models_pytorch.cosmos_pytorch import CosmosPi05Pytorch

logger = logging.getLogger("openpi")


@dataclasses.dataclass(frozen=True)
class CosmosPi05Config(pi0_config.Pi0Config):
    # HF hub id or local directory of the VLM checkpoint. Cosmos-Reason2 is gated on the hub: request access at
    # https://huggingface.co/nvidia/Cosmos-Reason2-8B and `huggingface-cli login` first. Any Qwen3-VL checkpoint
    # (e.g. "Qwen/Qwen3-VL-8B-Instruct") also works.
    vlm_path: str = "nvidia/Cosmos-Reason2-8B"

    # Square resolution fed to the Qwen3-VL vision tower. Must be a multiple of 32 (patch 16 x spatial merge 2).
    # Tokens per image = (size / 32) ** 2: 224 -> 49, 256 -> 64, 320 -> 100, 448 -> 196.
    # Qwen3-VL's processor never goes below 256x256 (min_pixels=65536), so 256 is the smallest in-distribution size.
    image_resolution: int = 256

    # --- What to train in the VLM ---
    # With everything frozen, the VLM prefix runs under no_grad (knowledge-insulation style); only the action expert
    # and the action/time projections are trained. This is the cheapest option and the recommended first run.
    freeze_vision: bool = True
    # Train the vision->LLM projection (Qwen3-VL "merger" + DeepStack mergers, ~60M params). Only meaningful if
    # freeze_vision is True (otherwise the whole vision tower is trained anyway).
    train_vision_merger: bool = False
    # LoRA rank on every linear layer of the LLM decoder (q/k/v/o/gate/up/down). 0 = LLM fully frozen.
    llm_lora_rank: int = 0
    llm_lora_alpha: float = 16.0
    # Fully fine-tune the last N decoder layers of the LLM (0 = frozen). The layers below stay frozen and run under
    # no_grad, so backprop stops at layer L-N. Note the action expert reads K/V from *every* layer, so this only adapts
    # the features it sees in the top N layers. In the last layer only input_layernorm/k_proj/k_norm/v_proj can affect
    # the loss (the expert consumes just its K/V), so the rest of that layer stays frozen. If LoRA is also enabled, it is
    # applied only to the layers that are not fully trained.
    train_llm_last_n_layers: int = 0
    # Learning-rate multiplier for every trainable VLM parameter (unfrozen layers, LoRA, merger/vision) relative to the
    # action expert and projections. Pretrained weights usually want a smaller LR than a from-scratch expert.
    vlm_lr_multiplier: float = 1.0

    # Qwen3-VL is a causal LM, so keep the prefix (images + prompt) causal to stay in its training distribution.
    # pi0/pi05 use a bidirectional prefix because PaliGemma was trained as a prefix-LM.
    causal_prefix: bool = True

    # --- Action expert ---
    # Depth / heads / kv heads / head_dim are read from the VLM config (they must match for the shared attention).
    # Only the residual width and MLP size are free. Params ~= depth * (width * heads*head_dim * 2
    # + width * kv*head_dim * 2 + 3 * width * mlp + 6 * width^2 [adaRMS]). 768/2048 on the 8B -> ~560M.
    action_expert_width: int = 768
    action_expert_mlp_dim: int = 2048

    pi05: bool = True

    # Not used by this model (kept only because Pi0Config defines them).
    paligemma_variant: str = "gemma_2b"  # type: ignore[assignment]
    action_expert_variant: str = "gemma_300m"  # type: ignore[assignment]
    pytorch_compile_mode: str | None = None

    def __post_init__(self):
        if not self.pi05:
            raise ValueError("CosmosPi05Config only supports pi05=True.")
        if self.train_llm_last_n_layers < 0:
            raise ValueError(f"train_llm_last_n_layers must be >= 0, got {self.train_llm_last_n_layers}")
        if self.image_resolution % 32 != 0:
            raise ValueError(f"image_resolution must be a multiple of 32, got {self.image_resolution}")
        super().__post_init__()

    @property
    @override
    def model_type(self) -> _model.ModelType:
        # Same data pipeline semantics as pi05 (quantile norm, discrete state in the prompt).
        return _model.ModelType.PI05

    @override
    def create(self, rng: at.KeyArrayLike):
        raise NotImplementedError("CosmosPi05Config is PyTorch-only. Train with scripts/train_pytorch.py.")

    def create_pytorch(self) -> "CosmosPi05Pytorch":
        from openpi.models_pytorch.cosmos_pytorch import CosmosPi05Pytorch

        return CosmosPi05Pytorch(self)

    @override
    def load_pytorch(self, train_config, weight_path: str):
        """Builds the model (loading the VLM from `vlm_path`) and loads the trained weights on top."""
        model = self.create_pytorch()
        model.load_trainable_weights(weight_path)
        return model

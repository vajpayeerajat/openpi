# Stage 1 
```
Loading checkpoint shards: 100%|██████████| 4/4 [00:00<00:00, 60.38it/s]
05:19:18.192 [I] CosmosPi05: trainable 583.6M / total 8728.4M params (action expert 582.3M, VLM trainable 0.0M, fully trained LLM layers: 0, grads from LLM layer 36) (155160:cosmos_pytorch.py:261)
05:19:22.105 [I] Enabled gradient checkpointing for memory optimization                           (155160:train_pytorch.py:634)
05:19:22.106 [I] Step 0 (after_model_creation): GPU memory - allocated: 18.68GB, reserved: 18.80GB, free: 0.12GB, peak_allocated: 18.68GB, peak_reserved: 18.80GB (155160:train_pytorch.py:512)
05:19:22.392 [I] Step 13500 (before_loading_checkpoint): GPU memory - allocated: 18.68GB, reserved: 18.80GB, free: 0.12GB, peak_allocated: 18.68GB, peak_reserved: 18.80GB (155160:train_pyto
```

## Stage 2
```
Skipping norm stats: False
data_config.repo_id: g1_pickplace
Loading checkpoint shards: 100%|██████████| 4/4 [00:00<00:00,  9.96it/s]
q16:04:04.193 [I] CosmosPi05: trainable 787.7M / total 8772.0M params (action expert 582.3M, VLM trainable 204.1M, fully trained LLM layers: 0, grads from LLM layer 0) (223773:cosmos_pytorch.py:261)
```

No. With the current `cosmos2_8b_lora_g1_pickplace` settings (`freeze_vision=True`, `train_vision_merger=True`, `llm_lora_rank=16`), here's what trains, based on `_setup_trainable` in [cosmos_pytorch.py:215-240](src/openpi/models_pytorch/cosmos_pytorch.py#L215-L240):

| Part of the VLM | Trained? |
|---|---|
| **Vision encoder** (the ViT blocks) | **Frozen.** `freeze_vision=True` skips the line that unfreezes `vlm.visual`. |
| Vision merger + DeepStack mergers (the vision→LLM projection, about 60M params) | **Trained, fully.** Lines 220–222. |
| **LLM decoder** base weights | **Frozen.** |
| LLM decoder LoRA adapters (rank 16 on q/k/v/o/gate/up/down in every layer) | **Trained.** Lines 234–240. |
| Embeddings, layer norms | Frozen. |
| Action expert + projections | Trained, warm-started from stage 1. |

So the decoder is adapted only through LoRA, and the vision encoder's features are fixed. Only the layer that maps them into the LLM is trained.

To train the vision encoder too, set `freeze_vision=False` (line 218–219 then unfreezes all of `vlm.visual`). That's roughly 0.5B more full-precision trainable parameters, plus their activations and Adam state, so expect a noticeable drop in speed and memory headroom. Batch 24 may not fit anymore. I'd run this LoRA + merger stage first, and only unfreeze vision if val stops improving again and the remaining errors look perception-related.
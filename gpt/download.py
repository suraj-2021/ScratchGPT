import os
import json
import torch
import numpy as np
from pathlib import Path

# we'll store the downloaded weights on disk
WEIGHTS_DIR  = Path("model_weights")
WEIGHTS_FILE = WEIGHTS_DIR / "gpt2_124m.pt"
CONFIG_FILE  = WEIGHTS_DIR / "gpt2_config.json"


#Check if weights already exist

def weights_already_downloaded():
    return WEIGHTS_FILE.exists() and CONFIG_FILE.exists()

# Download GPT-2 weights from HuggingFace

def download_gpt2_weights(force_download=False):
    if not force_download and weights_already_downloaded():
        print("[Downloader] Weights already on disk, skipping download.")
        print(f"             Found: {WEIGHTS_FILE}")
        return

    print("[Downloader]  Weights not found. Downloading from HuggingFace...")
    print("             Model: openai-community/gpt2 (124M parameters)")
    print("             This may take a minute — only happens once!\n")

    # Import HuggingFace transformers
    try:
        from transformers import GPT2Model as HF_GPT2
    except ImportError:
        raise ImportError(
            "Please install transformers: pip install transformers"
        )

    # Download model from HuggingFace
    hf_model = HF_GPT2.from_pretrained("openai-community/gpt2")
    hf_model.eval()
    hf_sd = hf_model.state_dict()

    print(f"[Downloader] HuggingFace model loaded. Remapping weights...")
    print(f"             HuggingFace has {len(hf_sd)} tensors to remap.\n")

    # Remap keys because:
    # HuggingFace uses  'transformer.h.0.attn.c_attn.weight'
    # Our model uses    'trf_blocks.0.attn.c_attn.weight'
    our_sd = _remap_hf_weights(hf_sd)

    # Save everything to disk
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(our_sd, WEIGHTS_FILE)

    # Save config too so we know what config was used
    config_info = {
        "source"       : "openai-community/gpt2",
        "vocab_size"   : 50257,
        "context_length": 1024,
        "emb_dim"      : 768,
        "n_heads"      : 12,
        "n_layers"     : 12,
    }
    with open(CONFIG_FILE, "w") as f:
        json.dump(config_info, f, indent=2)

    print(f"[Downloader]  Weights saved to: {WEIGHTS_FILE}")
    print(f"[Downloader]  Config  saved to: {CONFIG_FILE}")
    print("[Downloader]  All done! Future runs will skip the download.\n")


# HUGGING FACE WEIGHT REMAPPING

def _remap_hf_weights(hf_sd):
    our_sd = {}

    # Top-level embeddings
    our_sd["tok_emb.weight"] = hf_sd["wte.weight"]
    our_sd["pos_emb.weight"] = hf_sd["wpe.weight"]

    # Final layer norm
    our_sd["final_norm.weight"] = hf_sd["ln_f.weight"]
    our_sd["final_norm.bias"]   = hf_sd["ln_f.bias"]

    # ── Transformer blocks (0 to 11)
    for i in range(12):  # 12 layers for GPT-2 small
        hf_prefix  = f"h.{i}"         # HuggingFace prefix
        our_prefix = f"trf_blocks.{i}"  # our prefix

        # Layer norms
        our_sd[f"{our_prefix}.norm1.weight"] = hf_sd[f"{hf_prefix}.ln_1.weight"]
        our_sd[f"{our_prefix}.norm1.bias"]   = hf_sd[f"{hf_prefix}.ln_1.bias"]
        our_sd[f"{our_prefix}.norm2.weight"] = hf_sd[f"{hf_prefix}.ln_2.weight"]
        our_sd[f"{our_prefix}.norm2.bias"]   = hf_sd[f"{hf_prefix}.ln_2.bias"]

        # Attention weights
        # c_attn: projects input to Q, K, V
        our_sd[f"{our_prefix}.attn.c_attn.weight"] = \
            hf_sd[f"{hf_prefix}.attn.c_attn.weight"].T
        our_sd[f"{our_prefix}.attn.c_attn.bias"] = \
            hf_sd[f"{hf_prefix}.attn.c_attn.bias"]

        # c_proj: merges heads back
        our_sd[f"{our_prefix}.attn.c_proj.weight"] = \
            hf_sd[f"{hf_prefix}.attn.c_proj.weight"].T
        our_sd[f"{our_prefix}.attn.c_proj.bias"] = \
            hf_sd[f"{hf_prefix}.attn.c_proj.bias"]

        # Feed-forward (MLP) weights
        our_sd[f"{our_prefix}.ff.layers.0.weight"] = \
            hf_sd[f"{hf_prefix}.mlp.c_fc.weight"].T
        our_sd[f"{our_prefix}.ff.layers.0.bias"] = \
            hf_sd[f"{hf_prefix}.mlp.c_fc.bias"]
        our_sd[f"{our_prefix}.ff.layers.2.weight"] = \
            hf_sd[f"{hf_prefix}.mlp.c_proj.weight"].T
        our_sd[f"{our_prefix}.ff.layers.2.bias"] = \
            hf_sd[f"{hf_prefix}.mlp.c_proj.bias"]

    return our_sd



def load_gpt2_into_model(model, device="cpu"):

    # Download if needed (skips if file already on disk)
    download_gpt2_weights()

    print(f"[Loader] Loading weights from {WEIGHTS_FILE} ...")
    state_dict = torch.load(WEIGHTS_FILE, map_location=device, weights_only=True)

    # Load into model — strict=False in case there are minor mismatches
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    if missing:
        print(f"[Loader] Missing keys  ({len(missing)}): {missing[:3]}...")
    if unexpected:
        print(f"[Loader] Unexpected keys ({len(unexpected)}): {unexpected[:3]}...")

    model.to(device)
    model.eval()  # set to evaluation mode (disables dropout)

    print(f"[Loader] Weights loaded successfully! Model is on: {device}")
    return model


if __name__ == "__main__":
    from gpt.model import build_gpt2_model

    # Detect best available device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Using device: {device}\n")

    # Build the architecture (empty weights)
    model, cfg = build_gpt2_model()

    # Load GPT-2 pretrained weights (downloads once if not on disk)
    model = load_gpt2_into_model(model, device=device)

    # Quick forward pass to confirm it works
    dummy_input = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long).to(device)
    with torch.no_grad():
        logits = model(dummy_input)

    print(f"\n[Test] Input shape : {dummy_input.shape}")
    print(f"[Test] Output shape: {logits.shape}")  # should be [1, 5, 50257]
    print("[Test] Model working correctly!!")
import os
import torch
from pathlib import Path

# Load .env file — must happen before any os.environ.get() calls.
# If python-dotenv is not installed this silently does nothing.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from gpt.model import build_gpt2_model
from gpt.download import load_gpt2_into_model
from gpt.inference import GPT2Tokenizer


_model     = None
_tokenizer = None
_device    = None

BASE_DIR = Path(__file__).resolve().parent.parent
FINETUNED_WEIGHTS = BASE_DIR / "model_weights" / "gpt2_finetuned.pt"


def get_device():
    """Pick the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def initialise_model():
    """
    Called once at Django startup (in apps.py ready() method).
    Loads GPT-2 weights into the model and stores it globally.
    """
    global _model, _tokenizer, _device

    if _model is not None:
        return  # already loaded — do nothing

    _device    = get_device()
    _tokenizer = GPT2Tokenizer()

    print(f"\n[ModelLoader] Initialising GPT-2 on device: {_device}")

    # Build the architecture
    model, cfg = build_gpt2_model()

    import os
    print(f"[DEBUG] Files in model_weights: {os.listdir(BASE_DIR / 'model_weights')}")
    print(f"[DEBUG] Looking for: {FINETUNED_WEIGHTS}")

    if FINETUNED_WEIGHTS.exists():
        print(f"[ModelLoader] Fine-tuned weights found: {FINETUNED_WEIGHTS}")
        model    = load_gpt2_into_model(model, device=_device)
        ft_state = torch.load(FINETUNED_WEIGHTS, map_location=_device, weights_only=True)
        
        # --- NEW CODE: Translate Hugging Face keys to your custom keys ---
        custom_state_dict = {}
        for key, value in ft_state.items():
            new_key = key
            new_key = new_key.replace("transformer.wte.weight", "tok_emb.weight")
            new_key = new_key.replace("transformer.wpe.weight", "pos_emb.weight")
            new_key = new_key.replace("transformer.h.", "trf_blocks.")
            new_key = new_key.replace("transformer.ln_f.", "final_norm.")
            
            # Hugging Face uses Conv1D for attention/MLP, which requires transposing 
            # the weights to match standard PyTorch nn.Linear.
            if "attn.c_attn.weight" in key or "attn.c_proj.weight" in key or "mlp.c_fc.weight" in key or "mlp.c_proj.weight" in key:
                value = value.t()
                
            # Map MLP keys to your FeedForward keys
            new_key = new_key.replace("mlp.c_fc.", "ff.layers.0.")
            new_key = new_key.replace("mlp.c_proj.", "ff.layers.2.")
            
            custom_state_dict[new_key] = value
        # -----------------------------------------------------------------
        
        # Load the translated dictionary. strict=False is kept in case tied weights (like out_head) are missing from the dict.
        model.load_state_dict(custom_state_dict, strict=False)
        print("[ModelLoader] Fine-tuned weights loaded on top of GPT-2 base.")
    else:
        print("[ModelLoader] ℹ️  No fine-tuned weights found — using base GPT-2.")
        model = load_gpt2_into_model(model, device=_device)

    model.eval()
    _model = model
    print("[ModelLoader]  Model ready!\n")


def get_model():
    """Return the loaded model. Raises if not yet initialised."""
    if _model is None:
        raise RuntimeError(
            "Model not initialised. Call initialise_model() first, "
            "or ensure ChatConfig.ready() has been called."
        )
    return _model


def get_tokenizer():
    """Return the tokenizer instance."""
    if _tokenizer is None:
        raise RuntimeError("Tokenizer not initialised.")
    return _tokenizer


def get_device_str():
    """Return the device string."""
    return _device or get_device()
import os
import json
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

from gpt.model import build_gpt2_model
from gpt.download import load_gpt2_into_model
from gpt.inference import GPT2Tokenizer, format_chat_prompt



class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=256):
        self.examples = []
        skipped = 0
        print(f"[Dataset] Processing {len(data)} training examples...")

        for item in data:
            instruction = item.get("instruction", "").strip()
            response    = item.get("response", "").strip()
            if not instruction or not response:
                skipped += 1
                continue

            conversation = [{"role": "user",      "content": instruction},
                            {"role": "assistant", "content": response}]
            full_prompt = format_chat_prompt(conversation[:-1])
            full_text   = full_prompt + response + "\n\n"

            full_ids   = tokenizer.encode(full_text)
            prompt_ids = tokenizer.encode(full_prompt)

            if len(full_ids) > max_length:
                full_ids   = full_ids[:max_length]
                prompt_ids = prompt_ids[:max_length]

            labels  = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
            min_len = min(len(full_ids), len(labels))

            self.examples.append({
                "input_ids": full_ids[:min_len],
                "labels"   : labels[:min_len],
            })

        print(f"[Dataset] {len(self.examples)} examples ready ({skipped} skipped).")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def collate_fn(batch, pad_token_id=0):
    max_len = max(len(b["input_ids"]) for b in batch)
    ids_out, lbl_out = [], []
    for item in batch:
        pad = max_len - len(item["input_ids"])
        ids_out.append(item["input_ids"] + [pad_token_id] * pad)
        lbl_out.append(item["labels"]    + [-100]          * pad)
    return (
        torch.tensor(ids_out, dtype=torch.long),
        torch.tensor(lbl_out, dtype=torch.long),
    )


def load_alpaca_dataset(num_samples=2000):
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("pip install datasets")

    print(f"[Dataset] Downloading Alpaca ({num_samples} samples)...")
    ds   = load_dataset("tatsu-lab/alpaca", split="train")
    data = []
    for item in ds:
        if len(data) >= num_samples:
            break
        inst = item.get("instruction", "").strip()
        inp  = item.get("input",       "").strip()
        out  = item.get("output",      "").strip()
        if inp:
            inst = f"{inst}\n\nInput: {inp}"
        if inst and out:
            data.append({"instruction": inst, "response": out})
    print(f"[Dataset] Loaded {len(data)} examples.")
    return data


def load_custom_dataset(json_file_path):
    path = Path(json_file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {json_file_path}")
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    try:
        data = [json.loads(l) for l in content.splitlines() if l.strip()]
    except json.JSONDecodeError:
        data = json.loads(content)
    print(f"[Dataset] Loaded {len(data)} examples from {json_file_path}.")
    return data



def train_one_epoch(model, dataloader, optimizer, scaler, device, epoch_num, early_stop_loss=1.2):
    model.train()
    total_loss  = 0.0
    num_batches = len(dataloader)

    for batch_idx, (input_ids, labels) in enumerate(dataloader):
        input_ids = input_ids.to(device)
        labels    = labels.to(device)

        with torch.amp.autocast("cuda", enabled=(device == "cuda")):
            logits  = model(input_ids)
            B, T, V = logits.shape
            loss    = nn.CrossEntropyLoss(ignore_index=-100)(
                logits.view(B * T, V),
                labels.view(B * T)
            )

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        avg = total_loss / (batch_idx + 1)

        if (batch_idx + 1) % 10 == 0 or batch_idx == num_batches - 1:
            pct = (batch_idx + 1) / num_batches * 100
            print(f"  [Epoch {epoch_num}] Step {batch_idx+1}/{num_batches} "
                  f"({pct:.0f}%) | Loss: {avg:.4f}")

        # Early stopping — loss below target means overfitting is starting
        if avg < early_stop_loss:
            print(f"\n[EarlyStopping] Loss reached {avg:.4f} (below {early_stop_loss}) "
                  f"— stopping epoch early to prevent overfitting.")
            return avg

    return total_loss / num_batches



def fine_tune(
    data=None,
    dataset_path=None,
    num_samples=2000,
    epochs=1,               # 1 epoch is enough — overfitting risk is high
    batch_size=2,
    learning_rate=1e-5,     # gentle lr — prevents destroying GPT-2 knowledge
    max_length=256,         # 256 keeps full responses without truncation
    save_path="model_weights/gpt2_finetuned.pt",
    early_stop_loss=1.2,    # stop training when loss hits this value
    device=None,
):
    # Auto-detect device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\n[GPU] {gpu_name} ({vram_gb:.1f}GB VRAM) — mixed precision ON")

    print(f"\n{'='*60}")
    print(f"  GPT-2 INSTRUCTION FINE-TUNING")
    print(f"{'='*60}")
    print(f"  Device       : {device}")
    print(f"  Epochs       : {epochs}")
    print(f"  Batch size   : {batch_size}")
    print(f"  LR           : {learning_rate}")
    print(f"  Max length   : {max_length} tokens")
    print(f"  Early stop   : loss < {early_stop_loss}")
    print(f"  Target loss  : 1.2 – 1.8  (stop if below 1.2)")
    print(f"{'='*60}\n")

    # Load data
    if data is None:
        data = load_custom_dataset(dataset_path) if dataset_path \
               else load_alpaca_dataset(num_samples)

    tokenizer  = GPT2Tokenizer()
    dataset    = InstructionDataset(data, tokenizer, max_length=max_length)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, drop_last=True,
    )

    # Build model
    model, cfg = build_gpt2_model()
    model      = load_gpt2_into_model(model, device=device)

    # Enable dropout during fine-tuning (prevents memorisation)
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.1
    print("[FineTune] Dropout set to 0.1 for all layers.")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01,
        betas=(0.9, 0.95),
    )

    total_steps = len(dataloader) * epochs
    scheduler   = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=learning_rate * 0.1
    )

    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    best_loss  = float("inf")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        print(f"\n[Training] ── Epoch {epoch}/{epochs} ──")

        avg_loss = train_one_epoch(
            model, dataloader, optimizer, scaler,
            device, epoch, early_stop_loss=early_stop_loss
        )
        scheduler.step()

        elapsed = time.time() - epoch_start
        print(f"\n[Training] Epoch {epoch} done | "
              f"Avg Loss: {avg_loss:.4f} | "
              f"Time: {elapsed:.0f}s | "
              f"LR: {scheduler.get_last_lr()[0]:.2e}")

        if device == "cuda":
            used = torch.cuda.memory_allocated() / 1e9
            peak = torch.cuda.max_memory_allocated() / 1e9
            print(f"[GPU] VRAM used: {used:.2f}GB | Peak: {peak:.2f}GB")

        if avg_loss < best_loss:
            best_loss = avg_loss
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"[Training] New best model saved to: {save_path}")

    total_time = time.time() - start_time
    print(f"\n[Training] Done! Time: {total_time/60:.1f}min | Best loss: {best_loss:.4f}")
    print(f"[Training] Weights at: {Path(save_path).resolve()}")
    return model



if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data",       type=str,   default=None)
    p.add_argument("--samples",    type=int,   default=2000)
    p.add_argument("--epochs",     type=int,   default=1)
    p.add_argument("--batch",      type=int,   default=2)
    p.add_argument("--lr",         type=float, default=1e-5)
    p.add_argument("--maxlen",     type=int,   default=256)
    p.add_argument("--earlystop",  type=float, default=1.2)
    a = p.parse_args()

    fine_tune(
        dataset_path   = a.data,
        num_samples    = a.samples,
        epochs         = a.epochs,
        batch_size     = a.batch,
        learning_rate  = a.lr,
        max_length     = a.maxlen,
        early_stop_loss= a.earlystop,
    )
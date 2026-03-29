import torch
import tiktoken


class GPT2Tokenizer:
    def __init__(self):
        self._enc     = tiktoken.get_encoding("gpt2")
        self.vocab_size = self._enc.n_vocab
        self.BOS_TOKEN  = "<|endoftext|>"
        self.EOS_TOKEN  = "<|endoftext|>"
        self.BOS_ID     = self._enc.encode(self.BOS_TOKEN, allowed_special="all")[0]
        self.EOS_ID     = self.BOS_ID
        self.INST_START = "### Instruction:\n"
        self.RESP_START = "### Response:\n"

    def encode(self, text, allowed_special="all"):
        return self._enc.encode(text, allowed_special=allowed_special)

    def decode(self, token_ids):
        return self._enc.decode(token_ids)

    def encode_tensor(self, text, device="cpu"):
        ids = self.encode(text)
        return torch.tensor([ids], dtype=torch.long, device=device)


def format_chat_prompt(conversation_history, system_prompt=None):
    parts = []
    if system_prompt:
        parts.append(f"### System:\n{system_prompt}\n\n")
    for turn in conversation_history:
        role    = turn["role"]
        content = turn["content"].strip()
        if role == "user":
            parts.append(f"### Instruction:\n{content}\n")
        elif role == "assistant":
            parts.append(f"### Response:\n{content}\n\n")
    parts.append("### Response:\n")
    return "".join(parts)


@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt,
    max_new_tokens=256,
    temperature=0.8,
    top_k=40,
    repetition_penalty=1.2, # Added repetition penalty here
    device="cpu",
    stop_tokens=None,
):
    model.eval()

    input_ids   = tokenizer.encode_tensor(prompt, device=device)
    context_len = model.pos_emb.num_embeddings
    generated_ids = []
    stop_strings  = stop_tokens or []

    for step in range(max_new_tokens):
        current_ids = input_ids[:, -context_len:]
        logits      = model(current_ids)
        next_logits = logits[:, -1, :].clone()

        # --- Apply Repetition Penalty ---
        if repetition_penalty != 1.0:
            for token in set(generated_ids):
                if next_logits[0, token] < 0:
                    next_logits[0, token] *= repetition_penalty
                else:
                    next_logits[0, token] /= repetition_penalty

        # Temperature scaling
        if temperature != 1.0:
            next_logits = next_logits / temperature

        # Top-k filtering
        if top_k is not None and top_k > 1:
            values, _   = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
            threshold   = values[:, -1].unsqueeze(-1)
            next_logits = next_logits.masked_fill(next_logits < threshold, float('-inf'))

        probs   = torch.softmax(next_logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)

        next_token_int = next_id.item()

        # Stop on EOS
        if next_token_int == tokenizer.EOS_ID:
            break

        generated_ids.append(next_token_int)

        # Check stop strings
        generated_text_so_far = tokenizer.decode(generated_ids)
        for stop_str in stop_strings:
            if stop_str in generated_text_so_far:
                idx     = generated_text_so_far.find(stop_str)
                trimmed = generated_text_so_far[:idx]
                # Only return if there's real content before the stop string
                if trimmed.strip():
                    return trimmed.strip()
                # Stop string hit immediately with nothing before it —
                # strip the stop string and keep going
                generated_ids = tokenizer.encode(trimmed) if trimmed else []
                break

        input_ids = torch.cat([input_ids, next_id], dim=1)

    # Return whatever was generated — use strip() only to remove
    # leading/trailing whitespace, NOT to discard the whole response
    raw_tokens = generated_ids[:20]
    print(f"[TOKEN DEBUG] First 20 token IDs: {raw_tokens}")
    print(f"[TOKEN DEBUG] Decoded raw (no strip): '{tokenizer.decode(generated_ids[:50])!r}'")
    result = tokenizer.decode(generated_ids)
    return result.strip()


@torch.no_grad()
def generate_response(model, tokenizer, conversation_history, system_prompt=None, device="cpu", **gen_kwargs):
    prompt = format_chat_prompt(conversation_history, system_prompt=system_prompt)

    defaults = dict(
        max_new_tokens=300,
        temperature=0.8,
        top_k=50,
        repetition_penalty=1.2, # Added default penalty here as well
        device=device,
        stop_tokens=["### Instruction:"],
    )
    defaults.update(gen_kwargs)

    response = generate(model, tokenizer, prompt, **defaults)
    return response
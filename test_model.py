import torch
from gpt.model import build_gpt2_model
from gpt.download import load_gpt2_into_model
from gpt.inference import GPT2Tokenizer

device    = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = GPT2Tokenizer()

# 1. Build the model architecture
model, _  = build_gpt2_model()

# 2. Load the base GPT-2 weights
model     = load_gpt2_into_model(model, device=device)

# (Removed the fine-tuned weight loading here)

model.eval()

# 3. Setup the prompt
# Note: Since it's a base model, it might not respond perfectly to Instruct formats, 
# but it will attempt to complete the text.
prompt = "### Instruction:\nWhat is the capital of France?\n### Response:\n"
ids    = tokenizer.encode_tensor(prompt, device=device)

# 4. Generate Text
with torch.no_grad():
    for _ in range(80):
        logits  = model(ids)
        # Apply temperature scaling (0.8) and sample
        next_id = torch.multinomial(torch.softmax(logits[:, -1, :] / 0.8, dim=-1), 1)
        
        if next_id.item() == tokenizer.EOS_ID:
            break
            
        ids = torch.cat([ids, next_id], dim=1)

# 5. Decode and print
prompt_len = tokenizer.encode_tensor(prompt, device=device).shape[1]
generated  = tokenizer.decode(ids[0][prompt_len:].tolist())
print(f"\n=== GENERATED ===\n{repr(generated)}")
import torch
import torch.nn as nn

vocab_size = 10000
max_length = 512
batch_size = 32
sequence_length = 10
output_dim = 256

token_embeddings = nn.Embedding(vocab_size, output_dim)
input_tokens = torch.randint(0, vocab_size, (batch_size, sequence_length))
token_embeds = token_embeddings(input_tokens)

max_size = max_length
pos_embedding_vectors = nn.Embedding(max_size, output_dim)
pos_indices = torch.arange(sequence_length)
pos_embeddings = pos_embedding_vectors(pos_indices).unsqueeze(0).expand(batch_size, -1, -1)

input_embeddings = token_embeds + pos_embeddings
print("input_embeddings shape:", input_embeddings.shape)

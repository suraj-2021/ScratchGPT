import torch
import tiktoken
from torch.utils.data import Dataset, DataLoader 

#Extending the Dataset class

class GPTDatasetV1(Dataset):
    def __init__(self, text, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = [] 
        ids = tokenizer.encode(text)
        
        for i in range(0, len(ids) - max_length, stride):
            inputs = ids[i:i+max_length]
            targets = ids[i+1:i+max_length+1]
            
            self.input_ids.append(torch.tensor(inputs))
            self.target_ids.append(torch.tensor(targets))
            
    def __len__(self):
        return len(self.input_ids)
        
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
          
#building function to load data; hence dataloader
def create_dataloader(text, batch_size=4, max_length=4, stride=1, shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    
    dataset = GPTDatasetV1(text, tokenizer, max_length, stride)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
    
    return dataloader
    
#creating vector embedding to represent the vocabulary
torch.manual_seed(123)
vocab_size = 50256 
output_dimensions = 256

embedding_vector = torch.nn.Embedding(vocab_size, output_dimensions) 

#print the shape for demonstration purpose 
print(embedding_vector.weight.shape) 

#outputing the vectors from the embedding vector matrix who corrospond with the inputs
with open("myfile.txt", "r", encoding="utf8") as f:
    text = f.read()
dataloader = create_dataloader(text, batch_size=4, max_length=4, stride=4, shuffle=False, drop_last=True, num_workers=0)

initial_batch = next(iter(dataloader))
print(initial_batch)

#creating the positional embedding vector 
max_size = 4 
pos_embedding_vector = torch.nn.Embedding(max_size, output_dimensions)
pos_indices = torch.arange(max_size)
pos_embeddings = pos_embedding_vector(pos_indices)

#extrating data from dataloader 
inputs, targets = initial_batch 

#print the positional embeddings for demonstration purpose
print(pos_embeddings[:inputs.size(1)])
print(pos_embeddings[:targets.size(1)])

#final input embeddings
input_embeddings = embedding_vector(inputs) + pos_embeddings[:inputs.size(1)]
target_embeddings = embedding_vector(targets) + pos_embeddings[:targets.size(1)]

import re
import torch
import tiktoken
from torch.utils.data import Dataset, DataLoader

#creating gpt from scratch

#creating a naive vocubalary
with open("myfile.txt","r",encoding="utf8") as f:
     text = f.read()
tokens = re.split(r"([.,!?_-\"']|--|\s)",text)
tokens = [i.strip() for i in tokens if i.strip()]
tokens = set(tokens)
vocabulary = {j:i for i,j in enumerate(tokens)}

#implementing naive manual tokenization,encoding and decoding 
class GPTTokenizerV1:
      def __init__(self,vocabulary):
          self.char_to_int=vocabulary 
          self.int_to_char = {j:i for i,j in vocabulary.items()}
          
      def encode(self,text):
          preprocessed = re.split(r"([.,!?_-\"']|--|\s)",text)
          #remove spaces from "preprocessed"
          preprocessed=[i.strip() for i in preprocessed if i.strip()]
          
          #converting the preprocessed tokens into integer values 
          ids = [self.char_to_int[i] for i in preprocessed]
          
          return ids 
              
      def decode(self,ids):
          string =" ".join([self.int_to_char[c] for c in ids])
          return string
              
#using "tiktoken" over manual version for efficiency and flexibility
tokenizer = tiktoken.get_encoding("gpt2")
#encoding
ids = tokenizer.encode(text)
#decoding
string= tokenizer.decode(ids)

#manual class that generates the inputs and targets 
class InputTargetsGenerator:
      def __init__(self,text,tokenizer):
          self.inputs=[]
          self.targets = []
          self.ids = tokenizer.encode(text)
          self.generator()
          
      def generator(self):
          for i in range(0,len(self.ids),1):
              self.inputs.append(self.ids[i:i+4])
              self.targets.append(self.ids[i+1:i+4+1])
              
      def __getitem__(self,idx):
          return self.inputs[idx],self.targets[idx] 
          

#implementing Dataset and Dataloader to create dataset and load the data 
class GptDatasetV1(Dataset):
      def __init__(self,text,tokenizer,max_length,stride):
          self.input_ids=[]
          self.target_ids=[]
          
          ids = tokenizer.encode(text)
          
          for i in range(0,len(ids)-max_length,stride):
              self.input_ids.append(torch.tensor(ids[i:i+max_length]))
              self.target_ids.append(torch.tensor(ids[i+1:i+max_length+1]))
              
      def __len__(self):
          return len(self.input_ids)
     
      def __getitem__(self,idx):
          return self.input_ids[idx],self.target_ids[idx]
         
def create_dataloader(text,batch_size=4,max_length=4,stride=1,shuffle=False,drop_last=True,num_workers=0):
    
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GptDatasetV1(text,tokenizer,max_length,stride)
    
    dataloader = DataLoader(
            dataset,
            batch_size= batch_size,
            shuffle= shuffle,
            drop_last= drop_last,
            num_workers = num_workers
        )
        
    return dataloader 
    

#checking the dataloader
dataloader = create_dataloader(text,batch_size=4,max_length=4,stride=4,shuffle=False)
iter_dataloader = iter(dataloader)
first_batch = next(iter_dataloader)
print(first_batch)


#creating embedding_vector
vocab_size = 50256
output_dim = 256
embedding_vector = torch.nn.Embedding(vocab_size,output_dim)
#testing this embedding vector with first batch 
inputs,targets = first_batch
print(embedding_vector(inputs))
print(embedding_vector(targets))

#creating positional embeddings to counter position agnosticism of the transformers
max_length = 4
pos_embedding_vector = torch.nn.Embedding(max_length,output_dim) 
pos_embeddings = pos_embedding_vector(torch.arange(max_length)) 

#generating poisition aware final inputs and targets 

final_inputs = embedding_vector(inputs)+pos_embedding_vector(inputs) 
final_targets = embedding_vector(targets)+pos_embedding_vector(targets) 

#implementing initial steps of self attention mechanism 
#attention mechanism without trainable weights
query = inputs[0]
attn_scores_2 = torch.empty(inputs.shape[0])
for i,x_i in enumerate(inputs):
    attn_scores_2[i]= torch.dot(x_i.float(),query.float())

#now At this point We are aware That the above attn_scores_2 Variable stores values which are the resultant Of dot product between Query and all Elements inputs
#next step is to create attention weights with normalizing the attention scores 
attn_weights_2 = torch.softmax(attn_scores_2,dim=0)

# calculating the context vector z(2) by multiplying the
#embedded input tokens, x(i), with the corresponding attention weights and then summing the resulting vectors. Thus, context vector z(2) is the weighted sum of all input 
#vectors, obtained by multiplying each input vector by its corresponding attention weight
query = inputs[0]        
context_vec_2 = torch.zeros_like(query)
for i,x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i]*x_i.float()
print(context_vec_2)  

#now let's manually calculate the attention weights for all input tokens
attention_scores = torch.empty(inputs.shape[0],inputs.shape[0])
for i,x_i in enumerate(inputs):
    for j,x_j in enumerate(inputs):
        attention_scores[i,j]=torch.dot(x_i,x_j)

#since the use of for loops to calculate the attention scores is slower and computionally intensive task,we will use the matrix multiplication method 
attention_scores = inputs @ inputs.T     
#now let's find the attention weights of attention_scores 
attention_weights = torch.softmax(attention_scores,dim=-1)  

#now lets find the context vectors corrosponding to each token 
context_vectors = attention_weights @ inputs 





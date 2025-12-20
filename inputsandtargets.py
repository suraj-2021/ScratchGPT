import torch

class Dataset(torch.utils.data.IterableDataset):
    def __init__(self, data, block_size, batch_size, split="train"):
        super().__init__()
        self.data = data
        self.block_size = block_size
        self.batch_size = batch_size
        self.split = split

    def __iter__(self):
        while True:
            ix = torch.randint(
                len(self.data) - self.block_size - 1,
                (self.batch_size,)
            )
            x = torch.stack([self.data[i : i + self.block_size] for i in ix])
            y = torch.stack([self.data[i + 1 : i + self.block_size + 1] for i in ix])
            yield x, y

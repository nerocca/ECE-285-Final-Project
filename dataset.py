import torch
from torch.utils.data import Dataset

from tokenizer import DNATokenizer


class DNADataset(Dataset):

    def __init__(self, file, seq_len=1024):

        self.tokenizer = DNATokenizer()
        self.seq_len = seq_len

        with open(file) as f:
            self.lines = [x.strip() for x in f]

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):

        seq = self.lines[idx]

        ids = self.tokenizer.encode(seq)

        ids = ids[:self.seq_len]

        x = torch.tensor(ids[:-1])
        y = torch.tensor(ids[1:])

        return x, y
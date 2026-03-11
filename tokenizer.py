from transformers import PreTrainedTokenizer


class DNATokenizer:

    def __init__(self):

        self.vocab = {
            "A":0,
            "T":1,
            "C":2,
            "G":3,
            "N":4
        }

    def encode(self, seq):

        return [self.vocab[c] for c in seq if c in self.vocab]

    def decode(self, ids):

        inv = {v:k for k,v in self.vocab.items()}

        return "".join([inv[i] for i in ids])
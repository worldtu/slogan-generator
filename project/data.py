import torch
import pandas as pd
from torch.utils.data import Dataset
import torch.nn as nn

class CausalLMData(Dataset):
    """
    PyTorch Dataset for causal language modeling on desc→slogan sequences.
    """
    def __init__(self, csv_path: str, tokenizer, max_length: int = 128):
        df = pd.read_csv(csv_path)
        self.examples = []
        for _, row in df.iterrows():
            text = f"{tokenizer.bos_token} {row['desc']} {tokenizer.sep_token} {row['output']} {tokenizer.eos_token}"
            ids = tokenizer.encode(text)
            if len(ids) > max_length:
                ids = ids[:max_length]
                ids[-1] = tokenizer.eos_token_id
            self.examples.append(ids)
        self.pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ids = torch.tensor(self.examples[idx], dtype=torch.long)
        return ids[:-1], ids[1:]
    
    @staticmethod
    def collate_fn(batch):
        inputs, labels = zip(*batch)
        inputs = nn.utils.rnn.pad_sequence(inputs, batch_first=False, padding_value=0)
        labels = nn.utils.rnn.pad_sequence(labels, batch_first=False, padding_value=-100)
        return inputs, labels

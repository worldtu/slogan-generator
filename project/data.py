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
        
        # Define special token strings based on your TokenizerTrainer
        bos_token_str = "[CLS]"
        sep_token_str = "[SEP]"
        eos_token_str = "[SEP]" # Using SEP as EOS for this causal LM setup

        # Get token IDs
        # Ensure these tokens were part of 'special_tokens' in BpeTrainer
        bos_token_id = tokenizer.token_to_id(bos_token_str)
        sep_token_id = tokenizer.token_to_id(sep_token_str)
        eos_token_id = tokenizer.token_to_id(eos_token_str)
        self.pad_token_id = tokenizer.token_to_id("[PAD]")

        if bos_token_id is None or sep_token_id is None or eos_token_id is None or self.pad_token_id is None:
            raise ValueError("One or more special tokens ([CLS], [SEP], [PAD]) not found in tokenizer vocabulary. "
                             "Ensure they are in 'special_tokens' during BpeTrainer initialization.")

        for _, row in df.iterrows():
            text = f"{bos_token_str} {row['desc']} {sep_token_str} {row['output']} {eos_token_str}"
            encoding = tokenizer.encode(text)
            ids = encoding.ids # Get the list of token IDs
            
            if len(ids) > max_length:
                ids = ids[:max_length-1] + [eos_token_id] # Ensure sequence ends with EOS
            
            self.examples.append(ids)

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

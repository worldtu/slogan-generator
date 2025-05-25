import torch
import pandas as pd
from torch.utils.data import Dataset
import torch.nn as nn

class CausalLMData(Dataset):
    """
    PyTorch Dataset for causal language modeling on desc→slogan sequences.
    """
    def __init__(self, csv_path: str, tokenizer, max_length: int = 128, 
                str_start: str = "Desctiption: ", str_end: str = "Generate Slogan: "):
        df = pd.read_csv(csv_path)
        self.examples = []
        self.prompt_lengths = [] # Store prompt lengths
        self.str_start = str_start
        self.str_end = str_end

        for _, row in df.iterrows():
            # text = f"{tokenizer.bos_token} {self.str_start}{row['desc']} {self.str_end} {row['output']} {tokenizer.eos_token}"
            prompt_text = f"{tokenizer.bos_token} {self.str_start}{row['desc']} {self.str_end}" # Prompt part
            slogan_text = f"{row['output']} {tokenizer.eos_token}" # Slogan part, note leading space for consistency
            full_text = prompt_text + slogan_text

            prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False) # Don't add BOS/EOS here again
            self.prompt_lengths.append(len(prompt_ids))
            ids = tokenizer.encode(full_text, add_special_tokens=False) # Encode the whole thing
            
            if len(ids) > max_length:
                ids = ids[:max_length]
                if tokenizer.eos_token_id not in ids:
                    ids[-1] = tokenizer.eos_token_id
            
            self.examples.append(ids)

        self.pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ids = torch.tensor(self.examples[idx], dtype=torch.long)
        prompt_len = self.prompt_lengths[idx]
        return ids[:-1], ids[1:], prompt_len
    
    @staticmethod
    def collate_fn(batch):
        inputs, labels, prompt_lengths = zip(*batch)
        inputs = nn.utils.rnn.pad_sequence(inputs, batch_first=False, padding_value=0)
        labels = nn.utils.rnn.pad_sequence(labels, batch_first=False, padding_value=-100)
        prompt_lengths = torch.tensor(prompt_lengths, dtype=torch.long)
        return inputs, labels, prompt_lengths

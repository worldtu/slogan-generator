import torch
import pandas as pd
from torch.utils.data import Dataset

class CausalLMData(Dataset):
    """
    PyTorch Dataset for causal language modeling on desc→slogan sequences.
    """
    def __init__(self, csv_path: str, tokenizer, max_input_length: int = 128, max_target_length: int = 64):
        df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

        self.inputs = []
        self.targets = []
        for _, row in df.iterrows():
            # BART typically doesn't need BOS token for the encoder input if using from_pretrained model
            # The model.generate() method handles prepending bos_token_id for generation.
            # For labels, they should start with bos_token_id if the model expects it (usually handled internally or by shifting).
            # Let's keep it simple: tokenize source and target separately.
            
            source_text = str(row['desc'])
            target_text = str(row['output'])

            # Tokenize source (description)
            tokenized_input = self.tokenizer(
                source_text,
                max_length=self.max_input_length,
                padding="do_not_pad", # We'll pad in collate_fn
                truncation=True,
                return_tensors=None # Get list of IDs
            )
            
            # Tokenize target (slogan/output)
            # For BART, labels are the target_ids. The model internally shifts them for decoder_input_ids.
            tokenized_target = self.tokenizer(
                target_text,
                max_length=self.max_target_length,
                padding="do_not_pad", # We'll pad in collate_fn
                truncation=True,
                return_tensors=None # Get list of IDs
            )
            
            self.inputs.append(tokenized_input['input_ids'])
            self.targets.append(tokenized_target['input_ids'])

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.inputs[idx], dtype=torch.long),
            "labels": torch.tensor(self.targets[idx], dtype=torch.long)
        }
    
    def collate_fn(self, batch):
        # This method is now part of the DataLoader, not static if it needs self.tokenizer
        # For simplicity, we'll make it a static method and pass tokenizer's pad_token_id
        
        input_ids = [item['input_ids'] for item in batch]
        labels = [item['labels'] for item in batch]

        # Pad input_ids
        padded_inputs = self.tokenizer.pad(
            {"input_ids": input_ids},
            padding=True,
            return_tensors="pt",
        )
        
        # Pad labels
        padded_labels = self.tokenizer.pad(
            {"input_ids": labels}, # tokenizer.pad expects input_ids key
            padding=True,
            return_tensors="pt",
        )
        
        # For BART, label padding should be -100
        labels_input_ids = padded_labels['input_ids']
        labels_input_ids[labels_input_ids == self.tokenizer.pad_token_id] = -100
        
        return {
            "input_ids": padded_inputs['input_ids'],
            "attention_mask": padded_inputs['attention_mask'],
            "labels": labels_input_ids
        }
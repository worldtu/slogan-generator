# File: tokenizer.py
import os
import pandas as pd
from tokenizers import ByteLevelBPETokenizer
from transformers import PreTrainedTokenizerFast

class TokenizerTrainer:
    """
    Train and save a Byte-Pair Encoding tokenizer from raw CSV data.
    """
    def __init__(self, vocab_size: int = 32000, output_dir: str = "../tokenizer"):
        self.vocab_size = vocab_size
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def train(self, csv_path: str) -> PreTrainedTokenizerFast:
        """Train BPE tokenizer on desc + output columns and return HF tokenizer"""
        df = pd.read_csv(csv_path)
        texts = df["desc"].tolist() + df["output"].tolist()
        all_text_path = os.path.join(self.output_dir, "all_texts.txt")
        with open(all_text_path, "w", encoding="utf-8") as f:
            for line in texts:
                f.write(line + "\n")
        # Train the slow BPE tokenizer
        slow_tok = ByteLevelBPETokenizer()
        slow_tok.train(
            files=[all_text_path],
            vocab_size=self.vocab_size,
            min_frequency=2,
            special_tokens=["<bos>", "<sep>", "<eos>", "<unk>"]
        )
        # Save vocab and merges
        slow_tok.save_model(self.output_dir)
        # Build a fast tokenizer from the saved vocab and merges
        fast_tok = PreTrainedTokenizerFast(
            vocab_file=os.path.join(self.output_dir, "vocab.json"),
            merges_file=os.path.join(self.output_dir, "merges.txt"),
            bos_token="<bos>",
            sep_token="<sep>",
            eos_token="<eos>",
            unk_token="<unk>"
        )
        # Save the fast tokenizer files (tokenizer.json, config, etc.)
        fast_tok.save_pretrained(self.output_dir)
        return fast_tok
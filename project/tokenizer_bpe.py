from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import Lowercase, NFD, StripAccents, Sequence as NormalizerSequence
import pandas as pd
import os

class TokenizerTrainer:
    """
    Train and save a Byte-Pair Encoding tokenizer from raw CSV data.
    """
    def __init__(self, vocab_size: int = 10000, min_frequency=2, output_dir: str = "./tokenizer/"):
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        # 1. Initialize Tokenizer with a Model (BPE)
        self.tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        # 2. Pre-Tokenizer: Whitespace
        self.tokenizer.pre_tokenizer = Whitespace()
        # 3. Normalizer: Lowercase, NFD, StripAccents
        self.tokenizer.normalizer = NormalizerSequence([Lowercase(), NFD(), StripAccents()])
        # 4. Trainer
        self.trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]", "<company>", "[country]", "[country1]"], 
                                    vocab_size=vocab_size, min_frequency=min_frequency)
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def train(self, csv_path: str):
        """Train BPE tokenizer on desc + output columns and return HF tokenizer"""
        df = pd.read_csv(csv_path)
        texts = df["desc"].tolist() + df["output"].tolist()
        tokenizer_file_path = os.path.join(self.output_dir, "tokenizer.json")

        # Train the tokenizer
        self.tokenizer.train_from_iterator(texts, trainer=self.trainer)
        self.tokenizer.save(tokenizer_file_path)
        return self.tokenizer

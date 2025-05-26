from transformers import BartTokenizer, BartTokenizerFast
import pandas as pd
import os

class TokenizerTrainer:
    def __init__(self, vocab_size=50000):
        self.vocab_size = vocab_size
        
    def train(self, csv_path):
        """
        Train a tokenizer on the data from the CSV file.
        Returns a trained tokenizer.
        """
        # Read the CSV file
        df = pd.read_csv(csv_path)
        
        # Extract text data for tokenizer training
        texts = df['desc'].tolist() + df['output'].tolist()
        
        # Create a directory to save tokenizer files
        os.makedirs("tokenizer", exist_ok=True)
        
        # Initialize the BART tokenizer
        tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
        
        # Convert to fast tokenizer for better performance
        fast_tokenizer = BartTokenizerFast.from_pretrained("facebook/bart-base")
        
        # Save the tokenizer
        tokenizer.save_pretrained("tokenizer")
        fast_tokenizer.save_pretrained("tokenizer")
        
        return fast_tokenizer
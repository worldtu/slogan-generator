from transformers import BartTokenizerFast

def get_tokenizer(model_name_or_path: str = "distilbart-cnn-6-6"):
    """
    Loads a pre-trained BART tokenizer.
    """
    tokenizer = BartTokenizerFast.from_pretrained(model_name_or_path)
    return tokenizer
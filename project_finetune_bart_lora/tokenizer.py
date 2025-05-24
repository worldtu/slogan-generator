from transformers import BartTokenizerFast

def get_tokenizer(model_name_or_path: str):
    """
    Loads a pre-trained BART tokenizer.
    """
    tokenizer = BartTokenizerFast.from_pretrained(model_name_or_path)
    return tokenizer
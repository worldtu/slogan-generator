import torch

class SloganGenerator:
    """
    Generates slogans from a trained decoder-only model.
    """
    def __init__(self, model, tokenizer, device: str = "cpu"):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device

    def generate(self, desc: str, max_len: int = 128) -> str:
        self.model.eval()
        # Define special token strings based on your TokenizerTrainer
        bos_token_str = "[CLS]"
        sep_token_str = "[SEP]"
        eos_token_str = "[SEP]" # Assuming SEP is also used as EOS for generation stopping

        init = f"{bos_token_str} {desc} {sep_token_str}"
        ids = self.tokenizer.encode(init).ids # Get the list of token IDs

        out_ids = []
        with torch.no_grad():
            for _ in range(max_len):
                x = torch.tensor(ids, dtype=torch.long, device=self.device).unsqueeze(1)
                logits = self.model(x) # (T, 1, V)
                # Greedy decoding
                next_id = logits[-1, 0, :].argmax().item()
                ids.append(next_id)
                if next_id == self.tokenizer.token_to_id(eos_token_str): # Check against EOS token ID
                    break
                out_ids.append(next_id)
        
        return self.tokenizer.decode(out_ids)
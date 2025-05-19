import torch

class SloganGenerator:
    """
    Generates slogans from a trained decoder-only model.
    """
    def __init__(self, model, tokenizer, device: str = "cpu"):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device

    def generate(self, desc: str, max_len: int = 50) -> str:
        self.model.eval()
        init = f"{self.tokenizer.bos_token} {desc} {self.tokenizer.sep_token}"
        ids = torch.tensor(self.tokenizer.encode(init), dtype=torch.long, device=self.device).unsqueeze(1)
        for _ in range(max_len):
            logits = self.model(ids)
            next_id = logits[-1].argmax(dim=-1, keepdim=True)
            ids = torch.cat([ids, next_id.unsqueeze(0)], dim=0)
            if next_id.item() == self.tokenizer.eos_token_id:
                break
        seq = ids.squeeze(1).tolist()
        return self.tokenizer.decode(seq, skip_special_tokens=True)

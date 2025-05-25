import torch

class SloganGenerator:
    """
    Generates slogans from a trained decoder-only model.
    """
    def __init__(self, model, tokenizer, device: str = "cpu",
                str_start: str = "Desctiption: ", str_end: str = "Generate Slogan: "):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.str_start = str_start
        self.str_end = str_end

    def generate(self, desc: str, max_len: int = 32) -> str:
        self.model.eval()
        init = f"{self.tokenizer.bos_token} {self.str_start}{desc} {self.str_end}"
        ids = torch.tensor(self.tokenizer.encode(init), dtype=torch.long, device=self.device).unsqueeze(1)
        for _ in range(max_len):
            with torch.no_grad():
                logits = self.model(ids)
            next_id = logits[-1].argmax(dim=-1, keepdim=True)
            next_id = next_id.view(1, 1)
            ids = torch.cat([ids, next_id], dim=0)
            # if next_id.item() == self.tokenizer.eos_token_id:
            #     break
        seq = ids.squeeze(1).tolist()
        return self.tokenizer.decode(seq, skip_special_tokens=True)

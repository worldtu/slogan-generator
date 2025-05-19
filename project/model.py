import torch
import torch.nn as nn

class DecoderOnlyTransformer(nn.Module):
    """
    A decoder-only Transformer stack for causal language modeling.
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_head: int = 8,
        num_layers: int = 6,
        dim_ff: int = 2048,
        max_len: int = 512
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb   = nn.Embedding(max_len, d_model)
        layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=n_head, dim_feedforward=dim_ff)
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.out_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids):
        # input_ids: (T, B)
        T, B = input_ids.size()
        positions = torch.arange(T, device=input_ids.device).unsqueeze(1)
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        mem = torch.zeros(1, B, x.size(2), device=x.device)
        out = self.decoder(x, mem, tgt_mask=mask)
        logits = self.out_head(out)
        return logits
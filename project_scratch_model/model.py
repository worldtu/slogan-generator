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
        num_layers: int = 12,
        dim_ff: int = 2048,
        max_len: int = 512,
        dropout: int = 0.1
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb   = nn.Embedding(max_len, d_model)
        layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=n_head, dim_feedforward=dim_ff, 
                                            dropout=dropout, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.norm_f = nn.LayerNorm(d_model)
        self.out_head = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize parameters with better values
        self._init_parameters()

    def _init_parameters(self):
        """Initialize the parameters of the model with better values for training stability."""
        # Initialize embeddings
        nn.init.normal_(self.token_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)
        
        nn.init.ones_(self.norm_f.weight)
        nn.init.zeros_(self.norm_f.bias)

        # Initialize output projection
        nn.init.normal_(self.out_head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.out_head.bias)
        
        # Initialize transformer layers
        for name, param in self.decoder.named_parameters():
            if 'weight' in name and param.dim() > 1:
                # Initialize weights using Xavier uniform
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                # Initialize biases to zero
                nn.init.zeros_(param)

    def forward(self, input_ids):
        # input_ids: (T, B)
        T, B = input_ids.size()
        positions = torch.arange(T, device=input_ids.device).unsqueeze(1)
        
        # Token + positional embeddings
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        x = self.dropout(x)
        
        # Store original input for global residual connection
        original_input = x
        
        # Create causal mask for self-attention
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        # Memory for cross-attention (not used in decoder-only model)
        mem = torch.zeros(1, B, x.size(2), device=x.device)

        out = self.decoder(x, mem, tgt_mask=mask)
        # Add global residual connection
        out = out + original_input
        out = self.norm_f(out)

        logits = self.out_head(out)
        
        return logits
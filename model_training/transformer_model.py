import torch
import torch.nn as nn

class BrainToTextTransformer(nn.Module):
    def __init__(
        self,
        input_dim,
        vocab_size,
        d_model=256,
        nhead=8,
        num_layers=4,
        dim_feedforward=1024,
        dropout=0.1,
        max_len=2000,
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, d_model)

        # Learnable positional embedding
        self.pos_emb = nn.Parameter(torch.randn(1, max_len, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,   # IMPORTANT
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.classifier = nn.Linear(d_model, vocab_size)

    def forward(self, x, day_indicies=None, lengths=None):
        """
        x: (B, T, C)
        day_indicies: unused, kept for trainer compatibility
        """
        B, T, _ = x.shape

        x = self.input_proj(x)
        x = x + self.pos_emb[:, :T, :]

        if lengths is not None:
            mask = torch.arange(T, device=x.device)[None, :] >= lengths[:, None]
        else:
            mask = None

        x = self.encoder(x, src_key_padding_mask=mask)
        logits = self.classifier(x)

        return logits

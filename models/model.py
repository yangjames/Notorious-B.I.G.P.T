import pytorch_lightning as pl
import torch
from torch import Tensor, nn
from typing import Tuple


class TransformerBlock(nn.TransformerEncoderLayer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x):
        _, context_size, _ = x.shape
        attention_mask = torch.triu(
            torch.ones(context_size, context_size) * float("-inf"),
            diagonal=1
        ).to(x)
        out = super().forward(x, src_mask=attention_mask)
        return out


class NotoriousModel(pl.LightningModule):
    def __init__(
        self,
        context_length : int,
        vocab_size : int,
        num_embeddings : int = 64,
        num_heads : int = 16,
        num_transformer_layers : int = 1,
        feed_forward_dims : int = 2048,
        learning_rate : float = 1e-3,
    ):
        super(NotoriousModel, self).__init__()
        self.learning_rate = learning_rate
        self.context_length = context_length
        self.vocab_size = vocab_size
        self.num_embeddings = num_embeddings
        self.token_embedding_table = nn.Embedding(vocab_size, num_embeddings)
        self.position_embedding_table = nn.Embedding(
            context_length, num_embeddings)

        transformer_layers = []
        for i in range(num_transformer_layers):
            transformer_layers += [
                TransformerBlock(
                    d_model=num_embeddings,
                    nhead=num_heads,
                    dim_feedforward=feed_forward_dims,
                    batch_first=True
                )
            ]
        self.transformer_layers = nn.Sequential(*transformer_layers)
        self.head_layer = nn.Linear(num_embeddings, vocab_size)
        self.activation = nn.Softmax(dim=1)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x: Tensor) -> Tensor:
        batch_rows, context_size = x.shape
        # Convert word indices to embedding space.
        token_embeddings = self.token_embedding_table(x)
        position_embeddings = self.position_embedding_table(
            torch.arange(context_size).to(x)
        )
        # Token embeddings plus position embeddings.
        # B x context x embeddings
        x = token_embeddings + position_embeddings
        # Self attention
        x = self.transformer_layers(x)
        # From embedding space to vocabulary space logits.
        # Input is batch x context x embeddings
        # Output of linear layer is batch x context x vocab size
        x = self.head_layer(x)
        return x

    def training_step(self, batch : Tuple[Tensor, Tensor], batch_idx : int) -> Tensor:
        x, y = batch
        logits = self.forward(x)
        B, C, V = logits.shape
        logits = logits.view(B*C, V)
        y = y.view(B*C)
        loss = self.loss_fn(logits, y)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch : Tuple[Tensor, Tensor], batch_idx : int) -> Tensor:
        x, y = batch
        logits = self.forward(x)
        B, C, V = logits.shape
        logits = logits.view(B*C, V)
        y = y.view(B*C)
        loss = self.loss_fn(logits, y)
        self.log("valid/loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate)
        return optimizer

# Optional phoneme encoder placeholder.
import torch
import torch.nn as nn
import torch.nn.functional as F

class TinyPhonemeEncoder(nn.Module):
    def __init__(self, vocab: int = 64, emb_dim: int = 128):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb_dim)
        self.rnn = nn.GRU(emb_dim, emb_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(emb_dim*2, emb_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.emb(tokens)
        x, _ = self.rnn(x)
        x = x.mean(dim=1)
        x = self.fc(x)
        return F.normalize(x, dim=-1)

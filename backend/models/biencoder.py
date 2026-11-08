"""FairMatch Siamese Bi-Encoder Model (PyTorch).

Architecture:
  - Embedding layer (vocab_size x embed_dim=128)
  - Bidirectional LSTM (hidden_dim=64, so 128-d output per timestep)
  - Linear layer fc (hidden_dim*2 -> 64)
  - L2 normalization to produce unit-length embeddings

The same encoder tower is shared for both Resume and JD inputs
(Siamese / weight-tied architecture). Cosine similarity between
the two L2-normalised embeddings gives the match score.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------

class SiameseBiEncoder(nn.Module):
    """Shared-weight Siamese Bi-Encoder for resume–JD matching.

    Parameters
    ----------
    vocab_size : int
        Size of the token vocabulary (including PAD=0 and UNK=1).
    embed_dim : int
        Dimensionality of token embeddings. Default 128.
    hidden_dim : int
        LSTM hidden size *per direction*. Default 64.
        The bidirectional LSTM therefore outputs ``hidden_dim * 2`` features.
    pad_idx : int
        Padding token index in the vocabulary. Default 0.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 64,
        pad_idx: int = 0,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=pad_idx,
        )
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        # hidden_dim * 2 because bidirectional (128 -> 64)
        self.fc = nn.Linear(hidden_dim * 2, 64)

    # -- single-tower forward (shared weights) ----------------------------

    def forward_one(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Produce an L2-normalised embedding for a batch of token-id sequences.

        Parameters
        ----------
        token_ids : Tensor of shape ``(batch, seq_len)``

        Returns
        -------
        Tensor of shape ``(batch, 64)`` with unit L2-norm.
        """
        x = self.embedding(token_ids)                # (B, S, embed_dim)
        lstm_out, _ = self.lstm(x)                    # (B, S, hidden_dim*2)
        # Mean-pool over the sequence dimension
        pooled = lstm_out.mean(dim=1)                 # (B, hidden_dim*2)
        proj = self.fc(pooled)                        # (B, 64)
        normalised = F.normalize(proj, p=2, dim=1)
        return normalised

    # Alias for flexibility
    encode = forward_one

    def forward(
        self,
        resume_ids: torch.Tensor,
        jd_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode both inputs through the shared tower.

        Returns
        -------
        (resume_emb, jd_emb) – each of shape ``(batch, 64)``.
        """
        return self.forward_one(resume_ids), self.forward_one(jd_ids)


# ---------------------------------------------------------------------------
# Checkpoint helpers (CPU-safe as per project invariant)
# ---------------------------------------------------------------------------

_WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "weights"
_DEFAULT_MODEL_PATH = _WEIGHTS_DIR / "fair_ats_biencoder.pt"
_DEFAULT_META_PATH = _WEIGHTS_DIR / "ats_metadata.pkl"


def load_metadata(meta_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load the vocabulary and hyper-parameter metadata saved during Colab training.

    Expected keys in the pickle dict:
      - ``vocab`` : Dict[str, int] – token-to-index mapping
      - ``max_len`` : int – sequence pad/truncate length (128)
      - ``threshold`` : float – decision boundary (0.55)
    """
    meta_path = meta_path or _DEFAULT_META_PATH
    with open(meta_path, "rb") as f:
        metadata: Dict[str, Any] = pickle.load(f)
    return metadata


def load_model_checkpoint(
    model_path: Optional[Path] = None,
    meta_path: Optional[Path] = None,
) -> Tuple[SiameseBiEncoder, Dict[str, Any]]:
    """Instantiate :class:`SiameseBiEncoder` and load trained weights.

    All tensors are mapped to **CPU** (``map_location=torch.device('cpu')``)
    so the demo runs on any machine without CUDA.
    """
    metadata = load_metadata(meta_path)
    vocab_size = len(metadata["vocab"])

    model = SiameseBiEncoder(vocab_size=vocab_size)
    state_path = model_path or _DEFAULT_MODEL_PATH
    state_dict = torch.load(
        state_path,
        map_location=torch.device("cpu"),
        weights_only=True,
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model, metadata


def save_model_checkpoint(
    model: SiameseBiEncoder,
    metadata: Dict[str, Any],
    model_path: Optional[Path] = None,
    meta_path: Optional[Path] = None,
) -> None:
    """Persist model weights and metadata to disk."""
    model_path = model_path or _DEFAULT_MODEL_PATH
    meta_path = meta_path or _DEFAULT_META_PATH
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    with open(meta_path, "wb") as f:
        pickle.dump(metadata, f)


# ---------------------------------------------------------------------------
# Tokenisation helper (mirrors Colab training preprocessing)
# ---------------------------------------------------------------------------

def text_to_sequence(
    text: str,
    vocab: Dict[str, int],
    max_len: int = 128,
) -> torch.Tensor:
    """Convert sanitised text to a padded/truncated integer tensor.

    Tokenisation: lowercased whitespace split.
    Unknown tokens map to index 1 (``<UNK>``), padding to index 0 (``<PAD>``).

    Returns
    -------
    Tensor of shape ``(1, max_len)`` – batched for direct model input.
    """
    PAD_IDX = 0
    UNK_IDX = 1
    tokens = text.lower().split()
    ids = [vocab.get(tok, UNK_IDX) for tok in tokens[:max_len]]
    # Pad to max_len
    ids += [PAD_IDX] * (max_len - len(ids))
    return torch.tensor([ids], dtype=torch.long)


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("SiameseBiEncoder – Architecture Smoke Test")
    print("=" * 60)

    # Test loading actual Colab checkpoint if present
    if _DEFAULT_MODEL_PATH.exists() and _DEFAULT_META_PATH.exists():
        print(f"Loading checkpoint from {_DEFAULT_MODEL_PATH} ...")
        model, meta = load_model_checkpoint()
        print(f"[SUCCESS] Checkpoint loaded perfectly! Vocab size: {len(meta['vocab'])}, Threshold: {meta['threshold']}")
    else:
        DUMMY_VOCAB_SIZE = 500
        model = SiameseBiEncoder(vocab_size=DUMMY_VOCAB_SIZE)
        print("Using dummy model for architecture test:")
        print(model)

    dummy_resume = torch.randint(0, 100, (2, 128))
    dummy_jd = torch.randint(0, 100, (2, 128))
    r_emb, j_emb = model(dummy_resume, dummy_jd)
    print(f"Resume embedding shape : {r_emb.shape}") # (2, 64)
    print(f"JD embedding shape : {j_emb.shape}") # (2, 64)

    cos_sim = F.cosine_similarity(r_emb, j_emb)
    print(f"Cosine similarities : {cos_sim.tolist()}")

    # Verify L2-normalised
    norms = r_emb.norm(dim=1)
    print(f"L2 norms (should be 1) : {norms.tolist()}")
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), "L2 norm check failed!"

    # Verify forward_one matches encode
    f1_out = model.forward_one(dummy_resume)
    enc_out = model.encode(dummy_resume)
    assert torch.allclose(f1_out, enc_out), "forward_one and encode should match!"

    print("\n[SUCCESS] All checks passed successfully!")
    print("=" * 60)

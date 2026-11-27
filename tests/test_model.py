import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model import MultiHeadAttentionBlock, TransformerEncoderClassifier  # noqa: E402


def make_model(seq_len=16):
    torch.manual_seed(0)
    return TransformerEncoderClassifier(
        vocab_size=50, seq_len=seq_len, num_classes=2, d_model=32, N=2, h=4, d_ff=64
    ).eval()


def mask_for(ids, pad=0):
    return (ids != pad).unsqueeze(1).unsqueeze(1).int()


def test_output_shape():
    model = make_model()
    ids = torch.randint(1, 50, (3, 16))
    assert model(ids, mask_for(ids)).shape == (3, 2)


def test_padding_does_not_change_the_prediction():
    """Masked attention plus masked mean pooling: extra PAD tokens must be invisible."""
    model = make_model()
    tokens = torch.randint(1, 50, (2, 9))
    padded = torch.cat([tokens, torch.zeros(2, 7, dtype=torch.long)], dim=1)
    with torch.no_grad():
        short = model(tokens, mask_for(tokens))
        long = model(padded, mask_for(padded))
    torch.testing.assert_close(short, long, atol=1e-5, rtol=1e-5)


def test_attention_matches_pytorch_reference():
    torch.manual_seed(1)
    q, k, v = (torch.randn(2, 4, 6, 8) for _ in range(3))
    ours, weights = MultiHeadAttentionBlock.attention(q, k, v, mask=None, dropout=None)
    reference = F.scaled_dot_product_attention(q, k, v)
    torch.testing.assert_close(ours, reference, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(weights.sum(-1), torch.ones(2, 4, 6))


def test_masked_positions_get_no_attention():
    torch.manual_seed(2)
    q, k, v = (torch.randn(1, 2, 5, 4) for _ in range(3))
    mask = torch.tensor([1, 1, 1, 0, 0]).view(1, 1, 1, 5)
    _, weights = MultiHeadAttentionBlock.attention(q, k, v, mask=mask, dropout=None)
    assert weights[..., 3:].max().item() == pytest.approx(0.0, abs=1e-12)


def test_order_matters_thanks_to_positional_encoding():
    model = make_model()
    ids = torch.arange(1, 17).unsqueeze(0)
    with torch.no_grad():
        a = model(ids, mask_for(ids))
        b = model(ids.flip(1), mask_for(ids))
    assert not torch.allclose(a, b)


def test_heads_must_divide_model_width():
    with pytest.raises(AssertionError):
        MultiHeadAttentionBlock(d_model=30, h=4, dropout=0.0)

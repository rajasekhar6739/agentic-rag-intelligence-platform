from models.attention_lab import scaled_dot_product_attention
import torch

def test_attention_shapes():
    q = torch.randn(1, 4, 8)
    k = torch.randn(1, 4, 8)
    v = torch.randn(1, 4, 8)
    out, weights = scaled_dot_product_attention(q, k, v)
    assert out.shape == (1, 4, 8)
    assert weights.shape == (1, 4, 4)

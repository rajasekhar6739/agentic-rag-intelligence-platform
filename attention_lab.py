import torch
import torch.nn.functional as F

def scaled_dot_product_attention(q, k, v):
    d_k = q.size(-1)
    scores = q @ k.transpose(-2, -1) / (d_k ** 0.5)
    weights = F.softmax(scores, dim=-1)
    output = weights @ v
    return output, weights

if __name__ == "__main__":
    torch.manual_seed(7)
    q = torch.randn(1, 4, 8)
    k = torch.randn(1, 4, 8)
    v = torch.randn(1, 4, 8)
    output, weights = scaled_dot_product_attention(q, k, v)
    print("Output shape:", output.shape)
    print("Attention weights shape:", weights.shape)

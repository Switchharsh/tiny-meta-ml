import math
from transformers import GPT2Config, GPT2LMHeadModel

def make_model(total_params: int, vocab: int = 50257, seq: int = 256):
    """
    Returns GPT-2-like model with *exactly* `total_params` trainable weights.
    Solves for hidden size `h` from quadratic:
        12 h^2 + 4 v h = P
    """
    a, b = 12, 4 * vocab
    discriminant = b**2 + 4 * a * total_params
    h = max(32, int((-b + discriminant**0.5) / (2 * a)))  # ≥ 32
    n_layer = max(1, h // 64)
    n_head  = max(1, h // 64)
    cfg = GPT2Config(
        vocab_size=vocab,
        n_positions=seq,
        n_embd=h,
        n_layer=n_layer,
        n_head=n_head,
        tie_word_embeddings=True,
    )
    return GPT2LMHeadModel(cfg)
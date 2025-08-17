import numpy as np
import torch

def _to_bit_array(x):
    """
    Normalize x to a 1D numpy array of {0,1} (dtype=uint8).

    Accepts:
      - torch.Tensor (any shape, moved to cpu, squeezed)
      - numpy array (any shape, squeezed)
      - list/tuple
      - bitstring like "010101"
    Thresholding rules:
      - If in [-0.5, 0.5]: threshold at 0.0  (handles your 0→-0.5, 1→+0.5 encoding)
      - If in [0, 1]: threshold at 0.5
      - If in [-1, 1]: threshold at 0.0
      - Otherwise: treat as logits -> sigmoid then threshold at 0.5
    """
    # Strings → bits
    if isinstance(x, str):
        return np.array([1 if ch == '1' else 0 for ch in x], dtype=np.uint8)

    # Tensors → numpy
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()

    # Lists/tuples → numpy
    if isinstance(x, (list, tuple)):
        x = np.asarray(x)

    # Squeeze possible (B,L) or (1,L) etc. into 1-D
    x = np.squeeze(x)

    # Flatten if still >1D (e.g., (B,L) → (B*L,))
    if x.ndim > 1:
        x = x.reshape(-1)

    # Integers → just binarize
    if np.issubdtype(x.dtype, np.integer):
        x = (x != 0).astype(np.uint8)
        return x

    # Floats → threshold based on range
    x = x.astype(np.float32)
    xmin, xmax = float(x.min()), float(x.max())

    if -0.5 <= xmin and xmax <= 0.5:
        thr = 0.0
        bits = (x > thr).astype(np.uint8)
    elif 0.0 <= xmin and xmax <= 1.0:
        thr = 0.5
        bits = (x >= thr).astype(np.uint8)
    elif -1.0 <= xmin and xmax <= 1.0:
        thr = 0.0
        bits = (x > thr).astype(np.uint8)
    else:
        # Looks like logits: apply sigmoid safely, then 0.5 threshold
        x_clip = np.clip(x, -50.0, 50.0)
        sigm = 1.0 / (1.0 + np.exp(-x_clip))
        bits = (sigm >= 0.5).astype(np.uint8)

    return bits


def bit_accuracy(rec_bits, gt_bits, strict=False):
    """
    Compute bit accuracy (%) and BER given recovered and ground-truth messages.
    Returns: (acc_percent: float, ber: float, correct: int, total: int, used_len_equal: bool)
    - If lengths differ:
        - strict=True  -> raises ValueError
        - strict=False -> truncates to the shorter length
    """
    r = _to_bit_array(rec_bits)
    g = _to_bit_array(gt_bits)

    if r.size == 0 or g.size == 0:
        raise ValueError("Empty recovered or ground-truth message.")

    used_len_equal = (r.size == g.size)
    if not used_len_equal:
        if strict:
            raise ValueError(f"Length mismatch: recovered={r.size}, gt={g.size}")
        n = min(r.size, g.size)
        r, g = r[:n], g[:n]

    correct = int((r == g).sum())
    total = int(g.size)
    acc = 100.0 * correct / total
    ber = 1.0 - (acc / 100.0)
    return acc
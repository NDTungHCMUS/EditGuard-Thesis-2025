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

def split_bits_30(bitstr: str):
    """
    bitstr: chuỗi '0'/'1' (ví dụ: bit_input = sha256_bitstring(text_input))
    return: (chunks: list[str] mỗi chuỗi dài 30, last_real_len: int)
    """
    if not isinstance(bitstr, str):
        raise TypeError("bitstr must be a string of '0'/'1'")
    if any(c not in "01" for c in bitstr):
        raise ValueError("bitstr must contain only '0' and '1'")

    chunks = [bitstr[i:i+30] for i in range(0, len(bitstr), 30)]
    last_real_len = len(chunks[-1]) if chunks else 0
    if last_real_len == 0:
        return [], 0
    if last_real_len < 30:
        chunks[-1] = chunks[-1].ljust(30, '0')  # padding '0' đủ 30

    return chunks, 30 - last_real_len


def encode_ascii(text: str, errors: str = "strict") -> str:
    """
    Encode text to ASCII, then output a '0'/'1' bitstring (MSB-first, 8 bits/byte).
    errors: 'strict' | 'ignore' | 'replace'
    """
    b = text.encode("ascii", errors=errors)  # raises if non-ASCII and errors='strict'
    return ''.join(f'{byte:08b}' for byte in b)


def decode_ascii(bits: str, errors: str = "strict") -> str:
    """
    Decode a '0'/'1' bitstring (length must be multiple of 8) back to ASCII text.
    errors: passed to .decode() for safety, though ASCII should be exact.
    """
    if any(c not in '01' for c in bits):
        raise ValueError("bits must contain only '0' and '1'")
    if len(bits) % 8 != 0:
        raise ValueError("bitstring length must be a multiple of 8")

    byts = bytes(int(bits[i:i+8], 2) for i in range(0, len(bits), 8))
    return byts.decode("ascii", errors=errors)

import numpy as np

def split_into_tiles_128(img: np.ndarray, pad_mode: str = "edge"):
    """
    Split HxWxC (or HxW) image into 128x128 tiles.
    - Pads bottom/right so all tiles are exactly 128x128.
    - Returns: tiles (list of np.ndarray), coords (list of (y,x)),
               orig_hw ((H,W)), padded_hw ((Hp,Wp))
    """
    if img is None:
        raise ValueError("split_into_tiles_128: img is None")

    # Accept grayscale HxW → HxWx1
    if img.ndim == 2:
        img = img[..., None]
    if img.ndim != 3:
        raise ValueError(f"Expected HxWxC or HxW, got shape {img.shape}")

    H, W, C = img.shape
    tile = 128

    pad_h = (tile - (H % tile)) % tile
    pad_w = (tile - (W % tile)) % tile

    # Pad on bottom/right only
    img_padded = np.pad(
        img,
        pad_width=((0, pad_h), (0, pad_w), (0, 0)),
        mode=pad_mode  # 'edge'/'reflect'/'constant'
    )

    Hp, Wp, _ = img_padded.shape
    tiles, coords = [], []

    for y in range(0, Hp, tile):
        for x in range(0, Wp, tile):
            patch = img_padded[y:y+tile, x:x+tile, :]
            # make contiguous to avoid stride issues later
            tiles.append(np.ascontiguousarray(patch))
            coords.append((y, x))

    return tiles, coords, (H, W), (Hp, Wp)
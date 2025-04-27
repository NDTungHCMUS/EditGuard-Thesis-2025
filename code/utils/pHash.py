import cv2
import numpy as np

def pHash(image_path):
    # Step 1: Load and resize image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)

    # Step 2: Compute the DCT
    dct = cv2.dct(np.float32(img))

    # Step 3: Take top-left 8x8 of DCT
    dct_low_freq = dct[:8, :8]

    # Step 4: Compute median (excluding [0,0] DC term)
    dct_flat = dct_low_freq.flatten()
    median_val = np.median(dct_flat[1:])

    # Step 5: Generate hash (64 bits)
    hash_bits = ''.join(['1' if x > median_val else '0' for x in dct_flat])
    return hash_bits

# Example usage
hash1 = pHash("snitch.png")
hash2 = pHash("snitch_gen.png")

# Hamming distance
def hamming_distance(hash1, hash2):
    return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))

print("pHash1:", hash1)
print("pHash2:", hash2)
print("Hamming distance:", hamming_distance(hash1, hash2))

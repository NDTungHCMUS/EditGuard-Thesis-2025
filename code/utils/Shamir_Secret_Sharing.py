import random

# A prime larger than 2^64
PRIME = 18446744073709551629

def mod_inverse(a, p):
    """Compute the modular inverse of a modulo p using the Extended Euclidean Algorithm."""
    if a == 0:
        raise ZeroDivisionError("Division by zero encountered while computing modular inverse")
    lm, hm = 1, 0
    low, high = a % p, p
    while low > 1:
        r = high // low
        nm, new = hm - lm * r, high - low * r
        lm, low, hm, high = nm, new, lm, low
    return lm % p

def split_secret(secret, n, k, prime=PRIME):
    """
    Split a secret (an integer) into n shares with a threshold of k.
    Returns a list of (x, y) tuples representing the shares.
    """
    # Construct a random polynomial of degree k-1 with constant term equal to secret.
    coeffs = [secret] + [random.randrange(0, prime) for _ in range(k - 1)]
    shares = []
    for i in range(1, n + 1):
        x = i
        # Evaluate the polynomial at x.
        y = sum([coeff * pow(x, power, prime) for power, coeff in enumerate(coeffs)]) % prime
        shares.append((x, y))
    return shares

def recover_secret(shares, prime=PRIME):
    """
    Recover the secret from a list of shares using Lagrange interpolation.
    Each share is a tuple (x, y). At least k shares are required.
    """
    secret = 0
    for j, (xj, yj) in enumerate(shares):
        numerator = 1
        denominator = 1
        for m, (xm, _) in enumerate(shares):
            if m != j:
                numerator = (numerator * (-xm)) % prime
                denominator = (denominator * (xj - xm)) % prime
        lagrange_coeff = (numerator * mod_inverse(denominator, prime)) % prime
        secret = (secret + yj * lagrange_coeff) % prime
    return secret

# Example usage with error introduction:
if __name__ == '__main__':
    # The original 64-bit secret given as a binary string.
    secret_bin = "0111100100000110011101011011101011101100100110010110000111001111"
    secret = int(secret_bin, 2)
    print("Original Secret (int):", secret)
    print("Original Secret (bin):", format(secret, '064b'))
    
    # Split the secret into n shares with threshold k.
    n = 5  # total number of shares
    k = 3  # minimum shares needed to recover the secret
    shares = split_secret(secret, n, k)
    print("\nGenerated Shares:")
    for s in shares:
        print(s)
    
    # --- Introduce Error in One Share ---
    # Choose one share (here, the share at index 1) and simulate an error.
    share_to_corrupt_index = 1
    x, y = shares[share_to_corrupt_index]
    # Add a random error (non-zero) to the y value
    error_val = random.randint(1, 1000)
    corrupted_share = (x, (y + error_val) % PRIME)
    print("\nCorrupted share at index {}: {}".format(share_to_corrupt_index, corrupted_share))
    
    # Create a new list of shares that includes the corrupted share.
    shares_with_error = shares.copy()
    shares_with_error[share_to_corrupt_index] = corrupted_share
    
    # --- Attempt Recovery ---
    # Select any k shares from the shares_with_error (they might include the corrupted share).
    recovery_shares = random.sample(shares_with_error, k)
    print("\nSelected shares for recovery:")
    for s in recovery_shares:
        print(s)
    
    recovered_secret = recover_secret(recovery_shares)
    print("\nRecovered Secret (int):", recovered_secret)
    print("Recovered Secret (bin):", format(recovered_secret, '064b'))
    
    if recovered_secret == secret:
        print("\nSuccess: The recovered secret matches the original secret.")
    else:
        print("\nFailure: The recovered secret does not match the original secret.")

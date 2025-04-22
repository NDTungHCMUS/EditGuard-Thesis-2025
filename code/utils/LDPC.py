import numpy as np
import random
import math

def generate_ldpc_matrices(k=64, m=64, row_weight=3):
    """
    Generates a sparse parity matrix P (size m x k) with given row weight
    and constructs the systematic parity-check matrix H = [P | I_m].
    """
    P = np.zeros((m, k), dtype=int)
    for i in range(m):
        ones_positions = random.sample(range(k), row_weight)
        P[i, ones_positions] = 1
    I_m = np.eye(m, dtype=int)
    H = np.concatenate((P, I_m), axis=1)
    return P, H

def ldpc_encode(data_bit_str, P):
    """
    Encode a 64-bit data string using the systematic LDPC encoder.
    Parity bits p = (P * d) mod 2. The full codeword is [data || parity].
    """
    if len(data_bit_str) != 64:
        raise ValueError("Data must be exactly 64 bits.")
    d = np.array([int(b) for b in data_bit_str], dtype=int)
    p = np.mod(P.dot(d), 2)
    parity_str = "".join(str(bit) for bit in p)
    return parity_str

def introduce_random_errors(bitstring, num_errors=1):
    """
    Introduces random bit flips (errors) into the provided bitstring.
    """
    bit_list = list(bitstring)
    indices = random.sample(range(len(bit_list)), num_errors)
    for i in indices:
        bit_list[i] = '1' if bit_list[i] == '0' else '0'
    return "".join(bit_list)

def ldpc_decode_bp(received_codeword, H, p_channel=0.05, max_iter=100):
    """
    Decode a 128-bit LDPC codeword using belief propagation (sum-product algorithm).
    Args:
        received_codeword (str): 128-bit string (data and parity bits).
        H (np.array): Parity-check matrix of size (m x n) with n = 128.
        p_channel (float): Assumed error probability for the channel.
        max_iter (int): Maximum number of decoding iterations.
    Returns:
        str: Decoded 128-bit codeword as a string.
    """
    m, n = H.shape[0], H.shape[1]
    if len(received_codeword) != n:
        raise ValueError("Codeword length must match H matrix dimensions.")
    
    # Convert received codeword into a numpy array of 0s and 1s.
    r = np.array([int(b) for b in received_codeword], dtype=int)
    
    # Compute channel LLR. For a Binary Symmetric Channel:
    # LLR = log((1-p_channel)/p_channel) for a received 0, and negative for a received 1.
    Lc = np.log((1 - p_channel) / p_channel)
    L_ch = (1 - 2 * r) * Lc  # If r[i] is 0, L_ch[i] = Lc; if 1, L_ch[i] = -Lc.
    
    # Initialize messages: variable-to-check (M_vc) and check-to-variable (M_cv)
    M_vc = np.zeros((m, n))
    M_cv = np.zeros((m, n))
    
    # For each edge (j,i) with H[j,i]==1, initialize M_vc[j,i] with the channel LLR.
    for i in range(n):
        for j in range(m):
            if H[j, i] == 1:
                M_vc[j, i] = L_ch[i]
    
    # Iterative belief propagation.
    for iteration in range(max_iter):
        # Check node update.
        for j in range(m):
            # Get variable node indices connected to check node j.
            indices = np.where(H[j, :] == 1)[0]
            for i in indices:
                prod = 1.0
                for i_prime in indices:
                    if i_prime != i:
                        # Compute the hyperbolic tangent value of half the incoming message.
                        prod *= np.tanh(M_vc[j, i_prime] / 2.0)
                # Clip the product to avoid numerical issues with arctanh.
                prod = np.clip(prod, -0.999999, 0.999999)
                M_cv[j, i] = 2.0 * np.arctanh(prod)
        
        # Variable node update: combine channel information with incoming check messages.
        L_total = np.zeros(n)
        for i in range(n):
            check_indices = np.where(H[:, i] == 1)[0]
            L_total[i] = L_ch[i] + np.sum(M_cv[check_indices, i])
        
        # Make hard decisions based on the total LLR.
        decoded = np.array([0 if llr >= 0 else 1 for llr in L_total])
        
        # Check if all parity-check equations are satisfied.
        syndrome = np.mod(H.dot(decoded), 2)
        if np.all(syndrome == 0):
            print(f"Converged in {iteration + 1} iterations")
            break
        
        # Update variable-to-check messages for next iteration.
        for i in range(n):
            check_indices = np.where(H[:, i] == 1)[0]
            for j in check_indices:
                M_vc[j, i] = L_total[i] - M_cv[j, i]
    
    return "".join(str(bit) for bit in decoded)

if __name__ == '__main__':
    # Set seeds for reproducibility.
    # random.seed(42)
    # np.random.seed(42)
    
    # Generate LDPC matrices for a (128,64) code.
    P, H = generate_ldpc_matrices(k=64, m=64, row_weight=3)
    
    # Ghi các ma trận vào file "ldpc_matrices.txt"
    with open("ldpc_matrices.txt", "w") as f:
        f.write("Matrix P:\n")
        np.savetxt(f, P, fmt="%d")
        f.write("\nMatrix H:\n")
        np.savetxt(f, H, fmt="%d")
    
    print("LDPC matrices (P and H) have been written to 'ldpc_matrices.txt'.")
    
    # Original 64-bit data.
    original_data = "0011010110111101001010001001000100110101011011101010011110010010"
    print("Original 64-bit Data:")
    print(original_data)
    
    # Encode the data.
    parity = ldpc_encode(original_data, P)
    codeword = original_data + parity
    print("\nEncoded Codeword (128 bits):")
    print(codeword)
    
    # Introduce random errors: for example, flip 33 random bits.
    corrupted_codeword = introduce_random_errors(codeword, num_errors=33)
    print("\nCorrupted Codeword (128 bits) with 33 bit errors:")
    print(corrupted_codeword)
    print("Positions of errors:", [i for i, (a, b) in enumerate(zip(codeword, corrupted_codeword)) if a != b])
    
    # Decode using the belief propagation decoder.
    recovered_codeword = ldpc_decode_bp(corrupted_codeword, H, p_channel=0.05, max_iter=100)
    print("\nRecovered Codeword (128 bits) with BP decoding:")
    print(recovered_codeword)
    
    # Extract and check the recovered 64-bit data.
    recovered_data = recovered_codeword[:64]
    print("\nRecovered 64-bit Data:")
    print(recovered_data)
    
    if recovered_data == original_data:
        print("\nSuccess: Recovered data matches the original data.")
    else:
        print("\nWarning: Recovered data does not match the original data.")
        print("Positions of errors:", [i for i, (a, b) in enumerate(zip(original_data, recovered_data)) if a != b])

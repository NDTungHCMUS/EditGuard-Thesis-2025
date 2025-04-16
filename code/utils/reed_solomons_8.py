# ----- VN Start -----
from reedsolo import RSCodec, ReedSolomonError
import random
def binary_string_to_list_integer_8(binary_string):
    """
    Convert a binary string into a list of integers,
    each representing one 8-bit symbol.
    If the string's length is not a multiple of 8, pad with '0's at the beginning.
    """
    padding = (8 - len(binary_string) % 8) % 8
    binary_string = '0' * padding + binary_string
    # Divide the string into 8-bit chunks
    symbols = [binary_string[i:i+8] for i in range(0, len(binary_string), 8)]
    # Convert each 8-bit chunk to an integer
    list_integer = [int(symbol, 2) for symbol in symbols]
    return list_integer

def list_integer_to_binary_string_8(data):
    """
    Convert a list of integers (each representing an 8-bit symbol)
    into a binary string.
    """
    return "".join(format(x, '08b') for x in data)

def compute_parity_8(data_bit_str):
    """
    Compute the parity for the original 64-bit data (8 symbols of 8-bit)
    using RSCodec with 8 parity symbols.
    
    Args:
        data_bit_str (str): A 64-bit binary string representing the original data.
        
    Returns:
        str: A 64-bit binary string representing the parity (8 symbols of 8-bit).
    """
    if len(data_bit_str) != 64:
        raise ValueError("Original data must be 64 bits.")
    original_symbols = binary_string_to_list_integer_8(data_bit_str)  # 8 symbols
    if len(original_symbols) != 8:
        raise ValueError("Original data must be represented by 8 symbols (64 bits).")
    # Initialize RSCodec with 8 parity symbols (each symbol is 8-bit by default)
    rs = RSCodec(nsym=8, c_exp=8)
    # Encode the data (convert the list to a bytearray for RSCodec)
    encoded = rs.encode(bytearray(original_symbols))
    # Convert the encoded codeword to a list of integers
    encoded_ints = list(encoded)
    # The parity is the last 8 symbols of the codeword
    parity = encoded_ints[len(original_symbols):]
    return list_integer_to_binary_string_8(parity)

def recover_original_8(corrupted_bit_str):
    """
    Recover the 128-bit codeword (16 symbols of 8-bit) that may have errors,
    using RSCodec with 8 parity symbols.
    
    Args:
        corrupted_bit_str (str): A 128-bit binary string representing the corrupted codeword.
        
    Returns:
        str: A 128-bit binary string of the corrected codeword.
        
    Raises:
        ReedSolomonError: If the decoding process fails.
    """
    if len(corrupted_bit_str) != 128:
        raise ValueError("Codeword must be 128 bits.")
    corrupted_symbols = binary_string_to_list_integer_8(corrupted_bit_str)  # 16 symbols
    if len(corrupted_symbols) != 16:
        raise ValueError("Codeword must contain 16 symbols (128 bits).")
    rs = RSCodec(nsym=8, c_exp=8)
    try:
        # rs.decode returns a tuple: (decoded_message, corrected_codeword, errata_positions)
        decoded_message, corrected_codeword, _ = rs.decode(bytearray(corrupted_symbols))
    except ReedSolomonError as e:
        return -1
        raise ReedSolomonError("Unable to recover codeword: " + str(e))
    return list_integer_to_binary_string_8(list(corrected_codeword))

# Example usage:
if __name__ == '__main__':
    # Original 64-bit data (example)
    original_data = "1010101111001101111011110000111100001010101010101100110010101010"  # must be 64 bits
    if len(original_data) != 64:
        raise ValueError("Example original_data must be exactly 64 bits.")
    
    # Compute parity from the original data
    parity_bits = compute_parity_8(original_data)
    print("Original Data (64 bits):")
    print(original_data)
    print("\nComputed Parity (64 bits):")
    print(parity_bits)
    
    # Construct the full codeword: data (64 bits) + parity (64 bits) = 128 bits
    full_codeword = original_data + parity_bits
    print("\nFull Codeword (128 bits):")
    print(full_codeword)
    
    # Giới thiệu lỗi ngẫu nhiên: thay đổi 5 bit bất kỳ trong codeword
    def introduce_random_errors(bitstring, num_errors=20):
        bit_list = list(bitstring)
        indices = random.sample(range(len(bit_list)), num_errors)
        for i in indices:
            bit_list[i] = '1' if bit_list[i] == '0' else '0'
        return "".join(bit_list)
    
    corrupted_codeword = introduce_random_errors(full_codeword, num_errors=5)
    print("\nCorrupted Codeword (128 bits) with 5 bit errors:")
    print(corrupted_codeword)
    print("Positions of errors:", [i for i, (a, b) in enumerate(zip(full_codeword, corrupted_codeword)) if a != b])
    
    # Attempt to recover the original codeword from the corrupted one
    try:
        recovered_codeword = recover_original_8(corrupted_codeword)
        print("\nRecovered Codeword (128 bits):")
        print(recovered_codeword)
    except ReedSolomonError as err:
        print("\nError during recovery:")
        print(err)

# ----- VN End -----
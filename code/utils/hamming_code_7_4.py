# ----- VN Start -----
import random

def encode_hamming74(nibble):
    """
    Encode a 4-bit binary string (nibble) into a 7-bit Hamming(7,4) codeword.
    Positions (1-indexed): p1, p2, d1, p3, d2, d3, d4.
    Parity bits are computed as:
        p1 = d1 XOR d2 XOR d4
        p2 = d1 XOR d3 XOR d4
        p3 = d2 XOR d3 XOR d4
    """
    if len(nibble) != 4:
        raise ValueError("Nibble must be 4 bits.")
    d1 = int(nibble[0])
    d2 = int(nibble[1])
    d3 = int(nibble[2])
    d4 = int(nibble[3])
    p1 = d1 ^ d2 ^ d4
    p2 = d1 ^ d3 ^ d4
    p3 = d2 ^ d3 ^ d4
    # Build codeword: p1 p2 d1 p3 d2 d3 d4
    codeword = f"{p1}{p2}{d1}{p3}{d2}{d3}{d4}"
    return codeword

def decode_hamming74(block):
    """
    Decode a 7-bit Hamming(7,4) codeword.
    Detects and corrects a single-bit error.
    Returns the recovered 4-bit data string.
    """
    if len(block) != 7:
        raise ValueError("Block must be 7 bits.")
    bits = list(block)
    # Compute syndrome bits:
    # s1 = p1 XOR d1 XOR d2 XOR d4  (positions 1,3,5,7)
    # s2 = p2 XOR d1 XOR d3 XOR d4  (positions 2,3,6,7)
    # s3 = p3 XOR d2 XOR d3 XOR d4  (positions 4,5,6,7)
    s1 = int(bits[0]) ^ int(bits[2]) ^ int(bits[4]) ^ int(bits[6])
    s2 = int(bits[1]) ^ int(bits[2]) ^ int(bits[5]) ^ int(bits[6])
    s3 = int(bits[3]) ^ int(bits[4]) ^ int(bits[5]) ^ int(bits[6])
    # Syndrome as binary number (s3 s2 s1) gives error position (1-indexed)
    syndrome = s3 * 4 + s2 * 2 + s1
    if syndrome != 0:
        error_index = syndrome - 1  # convert to 0-indexed
        # Correct the error by flipping the bit
        bits[error_index] = '1' if bits[error_index] == '0' else '0'
    # Extract the 4 data bits from positions 3,5,6,7 (0-indexed: 2,4,5,6)
    data = bits[2] + bits[4] + bits[5] + bits[6]
    return "".join(data)

def compute_parity_hamming_74(data_bit_str):
    """
    Tính toán parity cho dữ liệu gốc 64 bit (mã hóa từng nibble theo Hamming(7,4), 
    trích xuất các bit parity từ mỗi khối, nối lại thành chuỗi parity có 48 bit và
    nếu không đủ 64 bit thì thêm "0" vào cuối cho đủ 64 bit).
    
    Args:
        data_bit_str (str): Chuỗi 64 bit của dữ liệu gốc.
        
    Returns:
        str: Chuỗi 64 bit của phần parity (48 bit từ Hamming được nối với 16 "0").
    """
    if len(data_bit_str) != 64:
        raise ValueError("Dữ liệu gốc phải là 64 bit.")
    # Chia dữ liệu gốc thành 16 nibble (mỗi nibble 4 bit)
    nibbles = [data_bit_str[i:i+4] for i in range(0, 64, 4)]
    parity_bits = ""
    for nibble in nibbles:
        # Mã hóa nibble dùng Hamming(7,4)
        codeword = encode_hamming74(nibble)
        # Lấy 3 bit parity từ các vị trí 1,2 và 4 (1-indexed) hay (0,1,3) 0-indexed.
        parity_bits += codeword[0] + codeword[1] + codeword[3]
    # Sau khi xử lý, parity_bits có độ dài 48 bit; nếu cần 64 bit, thêm "0" ở cuối.
    if len(parity_bits) < 64:
        parity_bits += "0" * (64 - len(parity_bits))
    return parity_bits

def recover_original_hamming_74(corrupted_bit_str):
    """
    Khôi phục codeword 128 bit đã bị lỗi. Codeword được cấu thành bởi 64 bit dữ liệu gốc và
    64 bit parity (theo cấu trúc: [dữ liệu 64 bit || parity 64 bit]). Quá trình khôi phục dựa vào mã
    Hamming(7,4) từng khối nibble:
        - Với mỗi nibble: lấy 4 bit dữ liệu từ phần dữ liệu và 3 bit parity từ 48 bit đầu của phần parity.
        - Dựng lại khối 7 bit với thứ tự: p1, p2, d1, p3, d2, d3, d4.
        - Giải mã Hamming(7,4) để khôi phục nibble gốc (sửa lỗi đơn nếu có).
    Sau đó, tính lại phần parity từ dữ liệu đã khôi phục.
    
    Args:
        corrupted_bit_str (str): Chuỗi 128 bit (codeword bị nhiễu lỗi).
        
    Returns:
        str: Chuỗi 128 bit của codeword đã được sửa chữa (dữ liệu 64 bit + parity 64 bit).
    """
    if len(corrupted_bit_str) != 128:
        raise ValueError("Codeword phải là 128 bit.")
    # Tách codeword thành phần dữ liệu 64 bit và phần parity 64 bit.
    data_part = corrupted_bit_str[:64]
    parity_part = corrupted_bit_str[64:]
    
    corrected_data = ""
    # Phần parity thật sự được dùng để khôi phục là 48 bit đầu (3 bit cho mỗi nibble)
    for i in range(16):
        # Lấy nibble dữ liệu (4 bit)
        nibble = data_part[i*4:(i+1)*4]
        # Lấy 3 bit parity ứng với nibble i từ phần parity (48 bit đầu)
        parity_block = parity_part[i*3:(i+1)*3]
        if len(parity_block) != 3:
            raise ValueError("Phần parity không đủ cho nibble thứ " + str(i))
        # Dựng lại khối 7 bit theo thứ tự: p1, p2, d1, p3, d2, d3, d4
        block = parity_block[0] + parity_block[1] + nibble[0] + parity_block[2] + nibble[1] + nibble[2] + nibble[3]
        # Giải mã Hamming(7,4) khôi phục nibble gốc (có thể sửa lỗi đơn)
        corrected_nibble = decode_hamming74(block)
        corrected_data += corrected_nibble
    
    # Tính lại phần parity từ dữ liệu đã khôi phục
    corrected_parity = compute_parity_hamming_74(corrected_data)
    # Trả về codeword đã được khôi phục (dữ liệu 64 bit + parity 64 bit)
    return corrected_data

# Ví dụ sử dụng:
if __name__ == '__main__':
    # Dữ liệu gốc: chuỗi 64 bit
    original_data = "1010101111001101111011110000111100001010101010101100110010101010"
    if len(original_data) != 64:
        raise ValueError("Original data must be exactly 64 bits.")
    
    print("Original 64-bit Data:")
    print(original_data)
    
    # Tính toán phần parity 64 bit dựa trên Hamming(7,4)
    parity_64bit = compute_parity_hamming_74(original_data)
    print("\nComputed 64-bit Parity:")
    print(parity_64bit)
    
    # Tạo codeword đầy đủ theo định dạng: [64 bit dữ liệu || 64 bit parity]
    full_codeword = original_data + parity_64bit
    print("\nFull Codeword (128 bits):")
    print(full_codeword)
    
    # Giới thiệu lỗi ngẫu nhiên: thay đổi 5 bit bất kỳ trong codeword
    def introduce_random_errors(bitstring, num_errors=20):
        bit_list = list(bitstring)
        indices = random.sample(range(len(bit_list)), num_errors)
        for i in indices:
            bit_list[i] = '1' if bit_list[i] == '0' else '0'
        return "".join(bit_list)
    
    corrupted_codeword = introduce_random_errors(full_codeword, num_errors=20)
    print("\nCorrupted Codeword (128 bits) with 5 bit errors:")
    print(corrupted_codeword)
    print("Positions of errors:", [i for i, (a, b) in enumerate(zip(full_codeword, corrupted_codeword)) if a != b])
    
    # Khôi phục codeword bằng phương pháp Hamming từng nibble
    recovered_codeword = recover_original_hamming_74(corrupted_codeword)
    print("\nRecovered Codeword (64 bits):")
    print(recovered_codeword)
    
    # So sánh phần dữ liệu được khôi phục với dữ liệu gốc
    recovered_data = recovered_codeword
    if recovered_data == original_data:
        print("\nSuccess: Recovered data matches the original data.")
    else:
        print("\nWarning: Recovered data does not match the original data.")
        print("Positions of errors:", [i for i, (a, b) in enumerate(zip(original_data, recovered_data)) if a != b])

# ----- VN End -----
# ----- VN Start -----
import random

def encode_hamming12_8(byte):
    """
    Mã hóa một chuỗi 8 bit (byte) thành codeword 12 bit theo Hamming(12,8).
    Sắp xếp bit (1-indexed):
        Vị trí parity: 1, 2, 4, 8
        Vị trí dữ liệu: 3, 5, 6, 7, 9, 10, 11, 12.
    
    Các bit dữ liệu là: d1,d2,d3,d4,d5,d6,d7,d8
    Các bit parity được tính theo công thức:
        p1 = d1 XOR d2 XOR d4 XOR d5 XOR d7
        p2 = d1 XOR d3 XOR d4 XOR d6 XOR d7
        p3 = d2 XOR d3 XOR d4 XOR d8
        p4 = d5 XOR d6 XOR d7 XOR d8
    """
    if len(byte) != 8:
        raise ValueError("Byte phải có 8 bit.")
    d1 = int(byte[0])
    d2 = int(byte[1])
    d3 = int(byte[2])
    d4 = int(byte[3])
    d5 = int(byte[4])
    d6 = int(byte[5])
    d7 = int(byte[6])
    d8 = int(byte[7])
    p1 = d1 ^ d2 ^ d4 ^ d5 ^ d7
    p2 = d1 ^ d3 ^ d4 ^ d6 ^ d7
    p3 = d2 ^ d3 ^ d4 ^ d8
    p4 = d5 ^ d6 ^ d7 ^ d8
    # Dựng codeword 12 bit: p1, p2, d1, p3, d2, d3, d4, p4, d5, d6, d7, d8.
    codeword = f"{p1}{p2}{d1}{p3}{d2}{d3}{d4}{p4}{d5}{d6}{d7}{d8}"
    return codeword

def decode_hamming12_8(block):
    """
    Giải mã codeword 12 bit theo Hamming(12,8).
    Phát hiện và sửa lỗi đơn bit dựa vào tính toán syndrome.
    
    Các bước:
      - Tính các bit syndrome s1, s2, s3, s4 dựa trên các parity check:
          s1 = p1 XOR d1 XOR d2 XOR d4 XOR d5 XOR d7  (tương đương: bit[0] ^ bit[2] ^ bit[4] ^ bit[6] ^ bit[8] ^ bit[10])
          s2 = p2 XOR d1 XOR d3 XOR d4 XOR d6 XOR d7  (bit[1] ^ bit[2] ^ bit[5] ^ bit[6] ^ bit[9] ^ bit[10])
          s3 = p3 XOR d2 XOR d3 XOR d4 XOR d8         (bit[3] ^ bit[4] ^ bit[5] ^ bit[6] ^ bit[11])
          s4 = p4 XOR d5 XOR d6 XOR d7 XOR d8         (bit[7] ^ bit[8] ^ bit[9] ^ bit[10] ^ bit[11])
      - Gộp syndrome theo thứ tự (s4 s3 s2 s1) cho ra vị trí bit bị lỗi (nếu không có lỗi thì syndrome = 0).
    
    Sau đó trích xuất 8 bit dữ liệu từ các vị trí: 3,5,6,7,9,10,11,12 (0-indexed: 2,4,5,6,8,9,10,11).
    
    Nếu giá trị syndrome vượt quá số lượng bit trong codeword (tức > 12), điều đó có thể cho thấy lỗi không đơn
    (hoặc lỗi không thể sửa bằng Hamming(12,8)) và bit lỗi sẽ không được sửa.
    """
    if len(block) != 12:
        raise ValueError("Block phải có 12 bit.")
    bits = list(block)
    bits = [int(b) for b in bits]
    
    s1 = bits[0] ^ bits[2] ^ bits[4] ^ bits[6] ^ bits[8]  ^ bits[10]
    s2 = bits[1] ^ bits[2] ^ bits[5] ^ bits[6] ^ bits[9]  ^ bits[10]
    s3 = bits[3] ^ bits[4] ^ bits[5] ^ bits[6] ^ bits[11]
    s4 = bits[7] ^ bits[8] ^ bits[9] ^ bits[10] ^ bits[11]
    
    syndrome = s4 * 8 + s3 * 4 + s2 * 2 + s1  # syndrome từ 0 đến 15
    if syndrome != 0:
        # Chuyển syndrome (1-indexed) về chỉ số 0-indexed.
        error_index = syndrome - 1
        if error_index < len(bits):
            bits[error_index] = 1 - bits[error_index]
        else:
            print("Warning: Syndrome out of range (", syndrome, 
                  "). Cannot correct error in block:", block)
    
    # Trích xuất 8 bit dữ liệu từ các vị trí: 3,5,6,7,9,10,11,12 (indices 2,4,5,6,8,9,10,11)
    data = [str(bits[i]) for i in [2, 4, 5, 6, 8, 9, 10, 11]]
    return "".join(data)

def compute_parity_hamming_12_8(data_bit_str):
    """
    Tính toán phần parity cho dữ liệu gốc 64 bit bằng cách:
      - Chia dữ liệu gốc thành 8 khối (mỗi khối 8 bit).
      - Mỗi khối mã hóa theo Hamming(12,8) cho ra codeword 12 bit.
      - Trích xuất 4 bit parity (ở vị trí 1,2,4,8) từ mỗi codeword.
      - Nối các khối parity lại với nhau (8*4 = 32 bit), nếu chưa đủ 64 bit thì đệm thêm "0" vào cuối.
    
    Args:
      data_bit_str (str): Chuỗi 64 bit dữ liệu gốc.
      
    Returns:
      str: Chuỗi 64 bit của phần parity.
    """
    if len(data_bit_str) != 64:
        raise ValueError("Dữ liệu gốc phải là 64 bit.")
    # Chia thành 8 khối, mỗi khối 8 bit
    blocks = [data_bit_str[i:i+8] for i in range(0, 64, 8)]
    parity_bits = ""
    for block in blocks:
        codeword = encode_hamming12_8(block)  # codeword 12 bit
        # Trích xuất 4 bit parity: ở vị trí 1,2,4,8 (indices 0,1,3,7)
        parity_bits += codeword[0] + codeword[1] + codeword[3] + codeword[7]
    # Sau khi xử lý, parity_bits có độ dài 32 bit; nếu cần 64 bit, thêm "0" ở cuối.
    if len(parity_bits) < 64:
        parity_bits += "0" * (64 - len(parity_bits))
    return parity_bits

def recover_original_hamming_12_8(corrupted_bit_str):
    """
    Khôi phục codeword 128 bit (cấu trúc: [64 bit dữ liệu || 64 bit parity])
    đã bị nhiễu lỗi dựa trên Hamming(12,8) từng khối:
      - Dữ liệu gốc gồm 64 bit được chia thành 8 khối (mỗi khối 8 bit).
      - Phần parity thực sự sử dụng là 32 bit đầu (4 bit cho mỗi khối).
      - Dựng lại mỗi khối 12 bit theo thứ tự: 
            p1, p2, d1, p3, d2, d3, d4, p4, d5, d6, d7, d8.
      - Giải mã Hamming(12,8) để khôi phục khối 8 bit dữ liệu.
      - Sau đó, tính lại phần parity từ dữ liệu đã khôi phục.
    
    Args:
      corrupted_bit_str (str): Chuỗi 128 bit (codeword bị nhiễu lỗi).
      
    Returns:
      str: Chuỗi 64 bit dữ liệu đã được khôi phục.
    """
    if len(corrupted_bit_str) != 128:
        raise ValueError("Codeword phải là 128 bit.")
    
    data_part = corrupted_bit_str[:64]
    parity_part = corrupted_bit_str[64:]
    
    recovered_data = ""
    # Với mỗi khối: 8 bit dữ liệu và 4 bit parity (từ 32 bit đầu của phần parity)
    for i in range(8):
        data_block = data_part[i*8:(i+1)*8]
        parity_block = parity_part[i*4:(i+1)*4]  # chỉ dùng 32 bit đầu, mỗi khối 4 bit
        if len(parity_block) != 4:
            raise ValueError("Phần parity không đủ cho khối thứ " + str(i))
        # Dựng lại khối 12 bit: p1, p2, d1, p3, d2, d3, d4, p4, d5, d6, d7, d8.
        block = parity_block[0] + parity_block[1] + data_block[0] + parity_block[2] + data_block[1:4] + parity_block[3] + data_block[4:]
        if len(block) != 12:
            raise ValueError("Khối codeword không đủ 12 bit ở khối thứ " + str(i))
        corrected_block = decode_hamming12_8(block)
        recovered_data += corrected_block
    
    # Tính lại phần parity từ dữ liệu đã khôi phục (để đối chiếu nếu cần)
    corrected_parity = compute_parity_hamming_12_8(recovered_data)
    # Có thể trả về dữ liệu đã khôi phục (64 bit) hoặc codeword đầy đủ
    # Ở đây ta trả về dữ liệu đã khôi phục.
    return recovered_data

# Ví dụ sử dụng:
if __name__ == '__main__':
    # Dữ liệu gốc: chuỗi 64 bit
    original_data = "1100110110101101001010001110100110000010101010100110011101000101"
    if len(original_data) != 64:
        raise ValueError("Dữ liệu gốc phải là 64 bit.")
    
    print("Original 64-bit Data:")
    print(original_data)
    
    # Tính toán phần parity 64 bit dựa trên Hamming(12,8)
    parity_64bit = compute_parity_hamming_12_8(original_data)
    print("\nComputed 64-bit Parity:")
    print(parity_64bit)
    
    # Tạo codeword đầy đủ theo định dạng: [64 bit dữ liệu || parity 64 bit]
    full_codeword = original_data + parity_64bit
    print("\nFull Codeword (128 bits):")
    print(full_codeword)
    
    # Giới thiệu lỗi ngẫu nhiên: thay đổi 20 bit bất kỳ trong codeword
    def introduce_random_errors(bitstring, num_errors=20):
        bit_list = list(bitstring)
        indices = random.sample(range(len(bit_list)), num_errors)
        for i in indices:
            bit_list[i] = '1' if bit_list[i] == '0' else '0'
        return "".join(bit_list)
    
    corrupted_codeword = introduce_random_errors(full_codeword, num_errors=20)
    print("\nCorrupted Codeword (128 bits) with 20 bit errors:")
    print(corrupted_codeword)
    print("Positions of errors:", [i for i, (a, b) in enumerate(zip(full_codeword, corrupted_codeword)) if a != b])
    
    # Khôi phục codeword bằng phương pháp Hamming(12,8) từng khối
    recovered_data = recover_original_hamming_12_8(corrupted_codeword)
    print("\nRecovered 64-bit Data:")
    print(recovered_data)
    
    if recovered_data == original_data:
        print("\nSuccess: Recovered data matches the original data.")
    else:
        print("\nWarning: Recovered data does not match the original data.")
        print("Positions of errors:", [i for i, (a, b) in enumerate(zip(original_data, recovered_data)) if a != b])

# ----- VN End -----
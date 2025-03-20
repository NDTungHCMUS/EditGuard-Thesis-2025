from reedsolo import RSCodec, ReedSolomonError

def binary_string_to_list_integer(binary_string):
    # Đảm bảo độ dài là bội số của 8 bằng cách thêm '0' vào đầu nếu cần
    padding = (8 - len(binary_string) % 8) % 8  # Số bit cần thêm vào đầu (nếu cần)
    binary_string = '0' * padding + binary_string

    # Chia chuỗi thành các khối 8-bit
    byte_list = [binary_string[i:i+8] for i in range(0, len(binary_string), 8)]

    # Chuyển mỗi phần từ nhị phân sang số nguyên
    list_integer = [int(byte, 2) for byte in byte_list]

    return list_integer

def list_integer_to_binary_string(original_data):
    binary_string = "".join(format(byte, '08b') for byte in original_data)

    return binary_string

def compute_parity(str):
    original_bytes = binary_string_to_list_integer(str)
    # Khởi tạo RSCodec với 8 ký hiệu parity trên GF(2^8)
    rs = RSCodec(nsym=8, c_exp=8)
    # Mã hóa: tổng codeword gồm 8 byte data và 8 byte parity (16 byte)
    encoded = rs.encode(original_bytes)
    # Lấy phần parity (8 byte sau dữ liệu gốc)
    parity = encoded[len(original_bytes):]
    return list_integer_to_binary_string(parity)


def recover_original(str):
    """
    Khôi phục dữ liệu gốc từ codeword 128-bit (16 byte) đã bị lỗi.
    
    Args:
        corrupted_codeword (bytes): Codeword bị nhiễu lỗi (16 byte) với tối đa 4 lỗi ký hiệu.
        
    Returns:
        original_bytes (bytes): Dãy dữ liệu gốc khôi phục được (8 byte).
        
    Nếu quá trình giải mã thất bại, hàm sẽ ném ra ReedSolomonError.
    
    Ví dụ:
        original = bytes([...])  # 8 byte
        parity = compute_parity(original)
        full_codeword = original + parity  # 16 byte
        # Giả sử ta đảo 4 bit ngẫu nhiên trên full_codeword:
        corrupted = bytearray(full_codeword)
        # ... (chèn lỗi vào corrupted) ...
        recovered = recover_original(bytes(corrupted))
    """
    corrupted_codeword = binary_string_to_list_integer(str)
    rs = RSCodec(nsym=8, c_exp=8)
    try:
        # rs.decode trả về tuple (decoded_message, full_codeword)
        decoded_tuple = rs.decode(corrupted_codeword)
        original_bytes = decoded_tuple[0]
    except ReedSolomonError as e:
        raise ReedSolomonError("Không khôi phục được dữ liệu gốc: " + str(e))
    
    return list_integer_to_binary_string(original_bytes)


# ===========================
# Ví dụ minh họa sử dụng các hàm trên:
if __name__ == '__main__':
    import random
    # Sinh dữ liệu gốc 8 byte (64 bit)
    original_data = binary_string_to_list_integer("1010110001100110111101101001011010101100110011011110110100101101")
    original_bytes = bytes(original_data)
    print("Original bytes: ", " ".join(f"{b:02X}" for b in original_bytes))
    
    # Tính parity từ dữ liệu gốc
    parity = compute_parity(original_bytes)
    # print("Parity bytes: ", " ".join(f"{b:02X}" for b in parity))
    print("Parity binary: ", list_integer_to_binary_string(parity))
    
    # Tạo full codeword (16 byte)
    full_codeword = original_bytes + parity
    print("Full codeword: ", " ".join(f"{b:02X}" for b in full_codeword))
    
    # Giả lập lỗi: đảo tổng cộng 4 bit ngẫu nhiên trên full codeword
    corrupted = bytearray(full_codeword)
    error_positions = random.sample(range(len(corrupted)), 4)
    print("Error positions: ", error_positions)
    for pos in error_positions:
        # Đảo 1 bit ngẫu nhiên trong byte tại vị trí pos
        bit_to_flip = 1 << random.randint(0, 7)
        corrupted[pos] ^= bit_to_flip
    
    print("Corrupted codeword: ", " ".join(f"{b:02X}" for b in corrupted))
    
    # Khôi phục dữ liệu gốc từ codeword đã bị lỗi
    try:
        recovered = recover_original(bytes(corrupted))
        print("Recovered bytes: ", " ".join(f"{b:02X}" for b in recovered))
        print("Recovery successful:", recovered == original_bytes)
    except ReedSolomonError as e:
        print("Recovery failed:", e)

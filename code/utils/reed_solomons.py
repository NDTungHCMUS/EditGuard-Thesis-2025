# ----- VN Start -----
from reedsolo import RSCodec, ReedSolomonError
import random

def binary_string_to_list_integer_16(binary_string):
    """
    Chuyển đổi chuỗi nhị phân thành danh sách các số nguyên, mỗi số biểu diễn 1 symbol 16 bit.
    Nếu chuỗi không đủ bội số của 16, thêm '0' vào đầu.
    """
    padding = (16 - len(binary_string) % 16) % 16
    binary_string = '0' * padding + binary_string
    # Chia chuỗi thành các khối 16-bit
    symbols = [binary_string[i:i+16] for i in range(0, len(binary_string), 16)]
    # Chuyển từng khối thành số nguyên
    list_integer = [int(symbol, 2) for symbol in symbols]
    return list_integer

def list_integer_to_binary_string_16(data):
    """
    Chuyển danh sách số nguyên (mỗi số biểu diễn 1 symbol 16 bit) thành chuỗi nhị phân.
    """
    return "".join(format(x, '016b') for x in data)

def compute_parity(data_bit_str):
    """
    Tính toán parity cho dữ liệu gốc 64 bit (4 symbol 16-bit) sử dụng RSCodec với 4 ký hiệu parity.
    
    Args:
        data_bit_str (str): Chuỗi 64 bit của dữ liệu gốc.
        
    Returns:
        str: Chuỗi 64 bit của phần parity (4 symbol 16-bit).
    """
    if len(data_bit_str) != 64:
        raise ValueError("Dữ liệu gốc phải là 64 bit.")
    original_symbols = binary_string_to_list_integer_16(data_bit_str)  # 4 symbol
    if len(original_symbols) != 4:
        raise ValueError("Dữ liệu gốc phải được biểu diễn bởi 4 symbol (64 bit).")
    # RSCodec với 4 ký hiệu parity (mỗi ký hiệu 16 bit)
    rs = RSCodec(nsym=4, c_exp=16)
    # Mã hóa: full codeword gồm 4 symbol dữ liệu + 4 symbol parity = 8 symbol
    encoded = rs.encode(original_symbols)
    # Lấy phần parity (4 symbol sau dữ liệu gốc)
    parity = encoded[len(original_symbols):]
    return list_integer_to_binary_string_16(parity)

def recover_original(corrupted_bit_str):
    """
    Khôi phục codeword 128 bit (8 symbol, mỗi symbol 16 bit) đã bị lỗi sử dụng RSCodec với 4 ký hiệu parity.
    
    Args:
        corrupted_bit_str (str): Chuỗi 128 bit (codeword bị nhiễu lỗi).
        
    Returns:
        str: Chuỗi 128 bit của codeword đã được sửa chữa.
        
    Nếu giải mã thất bại, ném ReedSolomonError.
    """
    if len(corrupted_bit_str) != 128:
        raise ValueError("Codeword phải là 128 bit.")
    corrupted_symbols = binary_string_to_list_integer_16(corrupted_bit_str)  # 8 symbol
    if len(corrupted_symbols) != 8:
        raise ValueError("Codeword phải chứa 8 symbol (128 bit).")
    rs = RSCodec(nsym=4, c_exp=16)
    try:
        # decode trả về 3 giá trị: (decoded_message, corrected_codeword, errata_positions)
        decoded_message, corrected_codeword, _ = rs.decode(corrupted_symbols)
    except ReedSolomonError as e:
        return -1
        raise ReedSolomonError("Không khôi phục được codeword: " + str(e))
    return list_integer_to_binary_string_16(corrected_codeword)


# ===========================
# Ví dụ minh họa:
if __name__ == '__main__':
    # Dữ liệu gốc 64 bit (4 symbol 16-bit)
    original_data = "1010110001100110111101101001011010101100110011011110110100101101"
    print("Original data (64-bit):")
    print(original_data)
    
    # Tính parity (sẽ trả về 64 bit, 4 symbol)
    parity = compute_parity(original_data)
    print("\nComputed parity (64-bit):")
    print(parity)
    
    # Ghép lại thành full codeword 128 bit (8 symbol)
    full_codeword = original_data + parity
    print("\nFull codeword (128-bit):")
    print(full_codeword)
    
    # Tách codeword thành 8 symbol (mỗi symbol 16-bit)
    # symbols = [full_codeword[i:i+16] for i in range(0, 128, 16)]
    # print("\nOriginal symbols:")
    # for idx, sym in enumerate(symbols):
    #     print(f"Symbol {idx}: {sym}")
    
    # # Giả lập lỗi: chèn lỗi vào 2 symbol bằng cách đảo 1 bit ngẫu nhiên trong mỗi symbol
    # error_indices = random.sample(range(8), 2)
    # print("\nError indices (symbol indices):", error_indices)
    # corrupted_symbols = symbols.copy()
    # for idx in error_indices:
    #     block = list(corrupted_symbols[idx])  # danh sách ký tự của symbol (16 bit)
    #     bit_to_flip = random.randint(0, 15)
    #     block[bit_to_flip] = '1' if block[bit_to_flip] == '0' else '0'
    #     corrupted_symbols[idx] = "".join(block)
    # corrupted_codeword = "".join(corrupted_symbols)
    # print("\nCorrupted full codeword (128-bit):")
    # print(corrupted_codeword)
    
    # Khôi phục codeword đã bị lỗi
    try:
        recovered_codeword = recover_original("10101100011001101111011010010110101011001100110111101101001011010001000101010100101110010000110000111100000111101000011101110110")
        print("\nRecovered full codeword (128-bit):")
        print(recovered_codeword)
        # print("\nRecovery successful:", recovered_codeword == full_codeword)
    except ReedSolomonError as e:
        print("Recovery failed:", e)

# ----- VN End -----
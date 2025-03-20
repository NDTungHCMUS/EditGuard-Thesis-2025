import numpy as np
def load_pairs_from_file(file_path):
    """
    Đọc file nhị phân, mỗi cặp (copyright, metadata) sẽ được lưu vào danh sách.
    
    Args:
        file_path (str): Đường dẫn đến file chứa dữ liệu.

    Returns:
        list: Danh sách các cặp (copyright, metadata).
    """
    pairs = []
    
    with open(file_path, "r") as f:
        lines = [line.strip() for line in f.readlines()]  # Đọc tất cả dòng và loại bỏ khoảng trắng

    # Đảm bảo số dòng là chẵn (mỗi cặp có 2 dòng)
    if len(lines) % 2 != 0:
        raise ValueError("Số dòng trong file không hợp lệ, phải là bội số của 2.")

    # Ghép cặp từng dòng copyright với dòng metadata
    for i in range(0, len(lines), 2):
        copyright_line = lines[i]
        metadata_line = lines[i + 1]
        pairs.append((copyright_line, metadata_line))

    return pairs

def bit_string_to_messagenp(bit_string, batch_size=1):
    """
    Chuyển chuỗi bit (ví dụ: "0101...") thành mảng numpy với mỗi bit được chuyển:
      '0' -> -0.5, '1' -> 0.5.
    
    Args:
        bit_string (str): Chuỗi bit, ví dụ có độ dài 64.
        batch_size (int): Số lượng bản sao (batch) cần tạo.
        
    Returns:
        np.ndarray: Mảng có kích thước (batch_size, len(bit_string)).
    """
    mapping = {'0': -0.5, '1': 0.5}
    # Chuyển đổi từng ký tự theo mapping
    message = np.array([mapping[b] for b in bit_string], dtype=float)
    # Reshape về dạng (1, message_length)
    message = message.reshape(1, -1)
    # Nếu cần nhiều batch, nhân bản theo batch_size
    if batch_size > 1:
        message = np.tile(message, (batch_size, 1))
    return message
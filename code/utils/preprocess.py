import numpy as np

def load_copyright(file_path):
    """
    Đọc từng dòng của file và trả về list các dòng.

    Args:
        file_path (str): Đường dẫn đến file.

    Returns:
        List[str]: Danh sách các dòng từ file, không chứa ký tự newline.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = [line.rstrip('\n') for line in file]
    return lines
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

def tensor_to_binary_string(tensor):
    """
    Chuyển đổi tensor chứa các giá trị 0.0 hoặc 1.0 thành chuỗi nhị phân.
    
    Args:
        tensor (torch.Tensor): Tensor với các giá trị 0 hoặc 1, có bất kỳ hình dạng nào.
    
    Returns:
        str: Chuỗi gồm các chữ số '0' và '1', ví dụ "1000011...".
    """
    # Đưa tensor về CPU nếu cần và làm phẳng thành vector 1 chiều
    flat = tensor.cpu().view(-1)
    # Chuyển từng giá trị sang int và tạo chuỗi nhị phân
    binary_string = ''.join(str(int(round(val.item()))) for val in flat)
    return binary_string
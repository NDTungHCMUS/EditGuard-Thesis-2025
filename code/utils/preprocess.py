import numpy as np
from .reed_solomons import compute_parity

def load_copyright_metadata_from_files(file_path):
    results = []
    with open(file_path, 'r') as file:
        lines = file.read().splitlines()  # reads lines without newline characters
    
    i = 0
    while i < len(lines):
        # The first line of a block is the image number (e.g. "0001"). Skip it.
        if len(lines[i]) == 4 and lines[i].isdigit():
            i += 1
        else:
            # If the expected image number is not found, you may handle the error as needed.
            raise ValueError(f"Expected an image number at line {i+1}, got: {lines[i]}")
        
        # Check if there is at least one 64-bit string for this image.
        if i >= len(lines):
            break
        
        # The first 64-bit string is considered as copyright.
        current_copyright = lines[i]
        i += 1
        current_metadata = []
        
        # All subsequent lines until the next image number or end of file are metadata.
        while i < len(lines):
            if len(lines[i]) == 4 and lines[i].isdigit():
                break
            current_metadata.append(lines[i])
            i += 1
        
        results.append({
            "copyright": current_copyright,
            "metadata": current_metadata
        })
    
    return results

def compute_parity_from_list_copyright_metadata(list_copyright_metadata):
    result = []
    for i in range(len(list_copyright_metadata)):
        parity_data = {
            "copyright": compute_parity(list_copyright_metadata[i]["copyright"]),
            "metadata": []
        }
        for metadata in list_copyright_metadata[i]["metadata"]:
            parity_metadata = compute_parity(metadata)
            parity_data["metadata"].append(parity_metadata)
        result.append(parity_data)
    return result

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
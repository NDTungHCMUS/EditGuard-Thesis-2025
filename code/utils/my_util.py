import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
import os
import math
from .reed_solomons import compute_parity, recover_original
from .util import save_img, tensor2img, decoded_message_error_rate

# ----- VN Start -----
## Explaination: Load copyright and metadata from file -> Return list of dictionary ({'copyright': str, 'metadata': list})
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

## Explaination: Compute correction code for copyright and metadata -> Return list of dictionary ({'copyright': str, 'metadata': list})
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

## Explaination: Return message correspond to dictionary (copyright or metadata)
def compute_message(index, dict_copyright_metadata, dict_parity_copyright_metadata):
    if (index == 0):
        return dict_copyright_metadata['copyright']
    elif (index == 1):
        return dict_parity_copyright_metadata['copyright']
    elif (index % 2 == 0):
        return dict_copyright_metadata['metadata'][index // 2 - 1]
    return dict_parity_copyright_metadata['metadata'][index // 2 - 1]

## Explaination: Convert bit string to numpy type
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

## Explaination: Take copyright, metadata (before and after) from lists
def get_copyright_metadata_from_list(list_message, list_recmessage):
    copyright_before = list_message[0]
    copyright_after = list_recmessage[0]
    metadata_before = ""
    metadata_after = ""
    for i in range(2, len(list_message), 2):
        metadata_before = metadata_before + list_message[i]
    for i in range(2, len(list_recmessage), 2):
        metadata_after = metadata_after + list_recmessage[i]
    return copyright_before, copyright_after, metadata_before, metadata_after

## Explaination: Convert tensor to binary string
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

## Explaination: Split parent images into child images
def split_and_save_image_torch(image_path, output_folder="images", num_child_images = 4):
    """
    Chia ảnh thành grid_size x grid_size phần bằng nhau và lưu vào thư mục output_folder.
    
    Args:
        image_path (str): Đường dẫn đến ảnh.
        output_folder (str): Thư mục để lưu ảnh.
        grid_size (int): Số phần theo mỗi chiều (mặc định 6, tức 36 phần).
    
    Yêu cầu:
        - Chiều cao (H) và chiều rộng (W) của ảnh phải chia hết cho grid_size.
    """
    # Đọc ảnh với PIL và chuyển thành RGB
    img = Image.open(image_path).convert("RGB")
    # Chuyển ảnh thành tensor có shape (C, H, W)
    img_tensor = TF.to_tensor(img)
    grid_size = int(math.sqrt(num_child_images))
    
    C, H, W = img_tensor.shape
    if H % grid_size != 0 or W % grid_size != 0:
        raise ValueError(f"Chiều cao H={H} và chiều rộng W={W} của ảnh phải chia hết cho grid_size {grid_size}.")
    
    patch_H = H // grid_size
    patch_W = W // grid_size
    
    os.makedirs(output_folder, exist_ok=True)  # Tạo thư mục nếu chưa tồn tại
    
    count = 0
    for i in range(grid_size):
        for j in range(grid_size):
            patch = img_tensor[:, i * patch_H:(i + 1) * patch_H, j * patch_W:(j + 1) * patch_W]
            patch_img = TF.to_pil_image(patch)
            patch_img.save(os.path.join(output_folder, f"{count}.png"))
            count += 1

    print(f"Đã lưu {count} ảnh vào thư mục {output_folder}")

def split_all_images(input_folder="A", output_folder="B", num_child_images = 4, num_images = 10):
    """
    Duyệt qua tất cả các ảnh trong thư mục input_folder, chia nhỏ và lưu vào output_folder.

    Args:
        input_folder (str): Thư mục chứa ảnh gốc.
        output_folder (str): Thư mục để lưu ảnh đã cắt.
    """
    os.makedirs(output_folder, exist_ok=True)  # Đảm bảo thư mục đầu ra tồn tại

     # Lấy danh sách các file, sắp xếp theo thứ tự tăng dần dựa trên tên file
    files = sorted(os.listdir(input_folder))
    
    image_files = [f for f in files if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    
    if num_images is not None:
        image_files = image_files[:num_images]
    
    # Duyệt qua các file ảnh đã sắp xếp
    for filename in image_files:
        img_path = os.path.join(input_folder, filename)
        img_name, _ = os.path.splitext(filename)  # Lấy tên file không có đuôi
        save_dir = os.path.join(output_folder, img_name)  # Tạo thư mục riêng cho ảnh
        os.makedirs(save_dir, exist_ok=True)
        
        # Gọi hàm split (giả sử hàm này đã được định nghĩa)
        split_and_save_image_torch(img_path, save_dir, num_child_images=num_child_images)

## Explaination: Combine child tensors to parent tensor (4 dimensions)
def combine_torch_tensors_4d(list_container, num_images=None):
    # Nếu không truyền num_images, mặc định bằng độ dài của list_container
    if num_images is None:
        num_images = len(list_container)
    
    # Kiểm tra xem list_container có đủ tensor
    if len(list_container) < num_images:
        raise ValueError(
            f"Danh sách chỉ có {len(list_container)} tensor, "
            f"nhưng yêu cầu ghép {num_images} tensor."
        )
    
    # Kiểm tra num_images có phải là số chính phương hay không
    root = int(math.sqrt(num_images))
    if root * root != num_images:
        raise ValueError(f"num_images = {num_images} không phải số chính phương (4, 9, 16, 25, 36, ...).")
    
    # Nếu list_container có nhiều hơn num_images, chỉ lấy đúng num_images phần tử đầu
    list_container = list_container[:num_images]
    
    # Lấy shape từ tensor đầu tiên để đối chiếu
    B, C, H, W = list_container[0].shape
    
    # Kiểm tra tất cả tensor có cùng shape
    for idx, tensor in enumerate(list_container):
        if tensor.shape != (B, C, H, W):
            raise ValueError(
                f"Tensor tại index {idx} có shape {tensor.shape} "
                f"không khớp với tensor đầu tiên ({B, C, H, W})."
            )
    
    # Ta sẽ tạo danh sách chứa các ảnh lưới của từng mẫu trong batch
    grid_list = []
    
    # Duyệt qua từng mẫu trong batch
    for b in range(B):
        # Lấy ra "ảnh" thứ b từ mỗi tensor (mỗi ảnh có shape (C, H, W))
        images = [t[b] for t in list_container]
        
        rows = []
        # Tạo lần lượt từng hàng (mỗi hàng chứa root ảnh ghép ngang)
        for i in range(root):
            # Ghép ngang 6 ảnh (hoặc root ảnh) theo dim=2
            row = torch.cat(images[i * root : (i + 1) * root], dim=2)
            rows.append(row)
        
        # Ghép dọc các hàng theo dim=1
        grid_image = torch.cat(rows, dim=1)  # shape (C, root*H, root*W)
        
        # Thêm batch dimension cho ảnh lưới rồi append vào list
        grid_list.append(grid_image.unsqueeze(0))  # shape (1, C, root*H, root*W)
    
    # Ghép tất cả ảnh lưới của từng mẫu batch lại với nhau theo dim=0
    result = torch.cat(grid_list, dim=0)  # shape (B, C, root*H, root*W)
    
    return result

## Explaination: Split parent tensor to child tensors (4 dimensions)
def split_torch_tensors_4d(parent_container_grid, num_child_images=4):
    grid_size = int(math.sqrt(num_child_images))
    B, C, H_total, W_total = parent_container_grid.shape

    # Kiểm tra H_total và W_total có chia hết cho grid_size không
    if H_total % grid_size != 0 or W_total % grid_size != 0:
        raise ValueError(f"H_total ({H_total}) và W_total ({W_total}) phải chia hết cho grid_size {grid_size}.")

    patch_H = H_total // grid_size
    patch_W = W_total // grid_size
    patches = []

    # Duyệt qua từng hàng và cột của lưới
    for i in range(grid_size):
        for j in range(grid_size):
            patch = parent_container_grid[:, :, 
                     i * patch_H:(i + 1) * patch_H, 
                     j * patch_W:(j + 1) * patch_W]
            patches.append(patch)

    return patches

## Explaination: Write output information to file
def write_extracted_messages(parent_image_id, copyright_before,
                             copyright_after, metadata_before,
                             metadata_after, out_file_path):
    """
    Ghi ra file thông tin trích xuất cho một ảnh với định dạng:
    
    Image_ID: <parent_image_id, định dạng 4 chữ số>
    Copyright Before: <dãy 64 bit>
    Copyright After: <dãy 64 bit>
    Copyright_Bit_Error: <số bit khác nhau giữa copyright trước và sau chia cho 64>
    Copyright_Wrong_Position: <list các vị trí khác nhau>
    Metadata Before: <dãy metadata, mỗi 64 bit cách nhau bởi dấu |>
    Metadata After: <dãy metadata, mỗi 64 bit cách nhau bởi dấu |>
    Metadata_Bit_Error: <số bit khác nhau giữa metadata trước và sau chia cho tổng số bit>
    Metadata_Wrong_Position: <list các vị trí khác nhau>
    General_Bit_Error: <số bit khác nhau trong toàn bộ dãy (copyright + metadata) chia cho tổng số bit>
    ---------------------
    
    Args:
        parent_image_id (int): Số nhận dạng ảnh gốc (0, 1, 2, ...).
        copyright_before (str): Dãy 64 bit trước khi sửa.
        copyright_after (str): Dãy 64 bit sau khi sửa.
        metadata_before (str): Dãy metadata trước khi sửa (độ dài bội số của 64).
        metadata_after (str): Dãy metadata sau khi sửa (độ dài bội số của 64).
        out_file_path (str): Đường dẫn đến file output.
        
    Returns:
        float: Giá trị general_bit_error.
    """
    # Định dạng parent_image_id thành chuỗi 4 chữ số
    image_id_str = f"{parent_image_id:04d}"
    
    # Kiểm tra độ dài của copyright và metadata
    if len(copyright_before) != 64 or len(copyright_after) != 64:
        raise ValueError("Copyright trước và sau phải có đúng 64 bit.")
    
    if len(metadata_before) % 64 != 0 or len(metadata_after) % 64 != 0:
        raise ValueError("Metadata trước và sau phải là bội số của 64 bit.")
    
    # Tính bit error cho copyright
    copyright_diff_positions = [i for i in range(64) 
                                if copyright_before[i] != copyright_after[i]]
    copyright_diff_count = len(copyright_diff_positions)
    copyright_bit_error = copyright_diff_count / 64
    
    # Tính bit error cho metadata
    metadata_len = len(metadata_before)
    metadata_diff_positions = [i for i in range(metadata_len)
                               if metadata_before[i] != metadata_after[i]]
    metadata_diff_count = len(metadata_diff_positions)
    metadata_bit_error = metadata_diff_count / metadata_len
    
    # Tính general bit error cho toàn bộ dãy (copyright + metadata)
    combined_before = copyright_before + metadata_before
    combined_after = copyright_after + metadata_after
    combined_length = len(combined_before)  # = 64 + metadata_len
    combined_diff_count = sum(1 for i in range(combined_length)
                              if combined_before[i] != combined_after[i])
    general_bit_error = combined_diff_count / combined_length

    # Format metadata: chia thành các block 64 bit cách nhau bởi dấu "|"
    def format_metadata(md):
        return " | ".join([md[i:i+64] for i in range(0, len(md), 64)])
    
    metadata_before_formatted = format_metadata(metadata_before)
    metadata_after_formatted = format_metadata(metadata_after)
    
    # Ghi thông tin ra file
    with open(out_file_path, 'a') as f:
        f.write(f"Image_ID: {image_id_str}\n")
        f.write(f"Copyright Before: {copyright_before}\n")
        f.write(f"Copyright After: {copyright_after}\n")
        f.write(f"Copyright_Bit_Error: {copyright_bit_error}\n")
        f.write(f"Copyright_Wrong_Position: {copyright_diff_positions}\n")
        f.write(f"Metadata Before: {metadata_before_formatted}\n")
        f.write(f"Metadata After: {metadata_after_formatted}\n")
        f.write(f"Metadata_Bit_Error: {metadata_bit_error}\n")
        f.write(f"Metadata_Wrong_Position: {metadata_diff_positions}\n")
        f.write(f"General_Bit_Error: {general_bit_error}\n")
        f.write("---------------------\n")
    
    return general_bit_error
# ----- VN End -----
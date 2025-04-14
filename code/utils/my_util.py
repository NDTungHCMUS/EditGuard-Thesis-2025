# ----- VN Start -----
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
import os
import math
from .reed_solomons_16 import compute_parity_16, recover_original_16
from .reed_solomons_8 import compute_parity_8, recover_original_8
from .hamming_code import compute_parity_hamming_74, recover_original_hamming_74
from .util import save_img, tensor2img, decoded_message_error_rate

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
def compute_parity_from_list_copyright_metadata(list_copyright_metadata, type_correction_code = 1):
    result = []
    for i in range(len(list_copyright_metadata)):
        if (type_correction_code == 1):
            copyright = compute_parity_16(list_copyright_metadata[i]["copyright"])
        elif (type_correction_code == 2):
            copyright = compute_parity_8(list_copyright_metadata[i]["copyright"])
        elif (type_correction_code == 3):
            copyright = compute_parity_hamming_74(list_copyright_metadata[i]["copyright"])
        else:
            copyright = "0" * 64
        parity_data = {
            "copyright": copyright,
            "metadata": []
        }
        for metadata in list_copyright_metadata[i]["metadata"]:
            if (type_correction_code == 1):
                parity_metadata = compute_parity_16(metadata)
            elif (type_correction_code == 2):
                parity_metadata = compute_parity_8(metadata)
            elif (type_correction_code == 3):
                parity_metadata = compute_parity_hamming_74(metadata)
            else:
                parity_metadata = "0" * 64
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

## Explaination: Take copyright, metadata (before and after) from lists (without correction)
def get_copyright_metadata_from_list_without_correction(list_message, list_recmessage):
    copyright_before = list_message[0]
    copyright_after = list_recmessage[0]
    metadata_before = ""
    metadata_after = ""
    for i in range(2, len(list_message), 2):
        metadata_before = metadata_before + list_message[i]
    for i in range(2, len(list_recmessage), 2):
        metadata_after = metadata_after + list_recmessage[i]
    return copyright_before, copyright_after, metadata_before, metadata_after

## Explaination: Take copyright, metadata (before and after) from lists (with correction)
def get_copyright_metadata_from_list_with_correction(list_message, list_recmessage, type_correction_code = 1):
    copyright_before = list_message[0]
    metadata_before = ""
    for i in range(2, len(list_message), 2):
        metadata_before = metadata_before + list_message[i]

    num_child_images = len(list_message)
    list_input_to_correct = []
    # Build string to do reed-solomons
    for i in range(0, num_child_images, 2):
        list_input_to_correct.append(list_recmessage[i])
    for i in range(1, num_child_images, 2):
        list_input_to_correct[i//2] += list_recmessage[i]
    print("LIST SOLOMON:", list_input_to_correct)

    cnt_cannot_solve = 0
    metadata_after = ""
    for i in range(0, num_child_images // 2):
        if (type_correction_code == 1):
            a = recover_original_16(list_input_to_correct[i])
        elif (type_correction_code == 2):
            a = recover_original_8(list_input_to_correct[i])
        elif (type_correction_code == 3):
            a = recover_original_hamming_74(list_input_to_correct[i])
        else:
            a = -1
        if (a == -1):
            print("Cannot solve Reed Solomon")
            cnt_cannot_solve += 1
            if (i == 0):
                copyright_after = list_input_to_correct[i][:64]
            else:
                metadata_after = metadata_after + list_input_to_correct[i][:64]
        else: 
            if (i == 0):
                copyright_after = a[:64]
            else:
                metadata_after = metadata_after + a[:64]
    return copyright_before, copyright_after, metadata_before, metadata_after, cnt_cannot_solve

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
def split_and_save_image_torch(image_path, output_folder="images", 
                               num_child_on_width_size=2, num_child_on_height_size=2):
    """
    Split an image into equal patches and save them in the output folder.
    
    Args:
        image_path (str): Path to the image.
        output_folder (str): Folder to save the patches.
        num_child_on_width_size (int): Number of patches along the width (columns).
        num_child_on_height_size (int): Number of patches along the height (rows).
        
    Requirements:
        - The image height (H) and width (W) must be divisible by num_child_on_height_size and
          num_child_on_width_size respectively.
    """
    # Open the image with PIL and convert to RGB.
    img = Image.open(image_path).convert("RGB")
    # Convert image to tensor with shape (C, H, W)
    img_tensor = TF.to_tensor(img)
    
    C, H, W = img_tensor.shape
    
    # Check that the image dimensions are divisible by the grid sizes.
    if H % num_child_on_height_size != 0 or W % num_child_on_width_size != 0:
        raise ValueError(
            f"Image height H={H} and width W={W} must be divisible by "
            f"num_child_on_height_size={num_child_on_height_size} and num_child_on_width_size={num_child_on_width_size}."
        )
    
    # Compute the size of each patch.
    patch_H = H // num_child_on_height_size
    patch_W = W // num_child_on_width_size
    
    os.makedirs(output_folder, exist_ok=True)  # Create the output folder if it doesn't exist.
    
    count = 0
    # Loop over rows (height) and columns (width)
    for i in range(num_child_on_height_size):
        for j in range(num_child_on_width_size):
            patch = img_tensor[:, 
                               i * patch_H:(i + 1) * patch_H, 
                               j * patch_W:(j + 1) * patch_W]
            patch_img = TF.to_pil_image(patch)
            patch_img.save(os.path.join(output_folder, f"{count}.png"))
            count += 1

    print(f"Saved {count} patches into the folder {output_folder}")

def split_all_images(input_folder="A", output_folder="B", num_images=10, 
                     num_child_on_width_size=2, num_child_on_height_size=2):
    """
    Process all images in input_folder, split them into patches, 
    and save the patches (for each image separately) into output_folder.
    
    Args:
        input_folder (str): Folder containing the original images.
        output_folder (str): Folder where the split patches will be saved.
        num_images (int): Number of images to process (if None, process all).
        num_child_on_width_size (int): Number of patches along the width.
        num_child_on_height_size (int): Number of patches along the height.
    """
    os.makedirs(output_folder, exist_ok=True)  # Ensure the output folder exists
    
    # Get a sorted list of files from the input folder.
    files = sorted(os.listdir(input_folder))
    image_files = [f for f in files if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    
    if num_images is not None:
        image_files = image_files[:num_images]
    
    # Process each image file
    for filename in image_files:
        img_path = os.path.join(input_folder, filename)
        img_name, _ = os.path.splitext(filename)  # Get file name without extension.
        save_dir = os.path.join(output_folder, img_name)  # Create a separate folder per image.
        os.makedirs(save_dir, exist_ok=True)
        
        split_and_save_image_torch(
            img_path, 
            output_folder=save_dir, 
            num_child_on_width_size=num_child_on_width_size, 
            num_child_on_height_size=num_child_on_height_size
        )

## Explaination: Combine child tensors to parent tensor (4 dimensions)
def combine_torch_tensors_4d(list_container, num_child_on_width_size, num_child_on_height_size):
    """
    Ghép các tensor (shape: (B, C, H, W)) trong list_container thành một ảnh lưới cho mỗi mẫu trong batch.
    
    Thay vì dùng một số duy nhất (num_images) và yêu cầu số đó là số chính phương, 
    ta truyền vào số lượng con theo chiều rộng (num_child_on_width_size) và chiều cao (num_child_on_height_size).
    
    Ví dụ: Nếu bạn có 48 ảnh con, bạn có thể ghép thành lưới 8 cột và 6 hàng bằng cách:
        combine_torch_tensors_4d(list_container, num_child_on_width_size=8, num_child_on_height_size=6)
    
    Args:
        list_container (list): Danh sách các 4D torch.Tensor với shape (B, C, H, W)
        num_child_on_width_size (int): Số ảnh con ghép ngang (số cột).
        num_child_on_height_size (int): Số ảnh con ghép dọc (số hàng).
    
    Returns:
        torch.Tensor: Ảnh lưới kết hợp có shape (B, C, num_child_on_height_size*H, num_child_on_width_size*W)
    """
    # Tổng số ảnh con cần ghép
    num_children = num_child_on_width_size * num_child_on_height_size
    
    # Kiểm tra số lượng tensor có đủ không
    if len(list_container) < num_children:
        raise ValueError(
            f"Danh sách chỉ có {len(list_container)} tensor, nhưng yêu cầu ghép {num_children} tensor."
        )
    
    # Nếu có nhiều hơn, chỉ lấy đúng num_children phần tử đầu
    list_container = list_container[:num_children]
    
    # Lấy shape từ tensor đầu tiên để đối chiếu
    B, C, H, W = list_container[0].shape
    
    # Kiểm tra tất cả tensor có cùng shape
    for idx, tensor in enumerate(list_container):
        if tensor.shape != (B, C, H, W):
            raise ValueError(
                f"Tensor tại index {idx} có shape {tensor.shape} không khớp với tensor đầu tiên ({B}, {C}, {H}, {W})."
            )
    
    # Tạo danh sách chứa ảnh lưới cho từng mẫu trong batch
    grid_list = []
    for b in range(B):
        # Lấy ra ảnh thứ b từ mỗi tensor (mỗi ảnh có shape (C, H, W))
        images = [t[b] for t in list_container]
        
        rows = []
        # Ghép từng hàng: mỗi hàng chứa num_child_on_width_size ảnh liên tiếp
        for i in range(num_child_on_height_size):
            start_idx = i * num_child_on_width_size
            end_idx   = (i + 1) * num_child_on_width_size
            row = torch.cat(images[start_idx:end_idx], dim=2)  # Ghép theo chiều rộng
            rows.append(row)
        
        # Ghép dọc các hàng để tạo ảnh lưới cho mẫu thứ b
        grid_image = torch.cat(rows, dim=1)  # Kết quả có shape (C, num_child_on_height_size * H, num_child_on_width_size * W)
        grid_list.append(grid_image.unsqueeze(0))  # Thêm batch dimension
    
    # Ghép lại tất cả các ảnh lưới của từng mẫu trong batch theo dim=0
    result = torch.cat(grid_list, dim=0)  # Kết quả có shape (B, C, num_child_on_height_size * H, num_child_on_width_size * W)
    
    return result

## Explaination: Split parent tensor to child tensors (4 dimensions)
def split_torch_tensors_4d(parent_container_grid, num_child_on_width_size, num_child_on_height_size):
    """
    Tách một tensor 4D (đại diện cho một lưới ảnh) thành các "patch" (ảnh con)
    theo số lượng ảnh con theo chiều rộng và chiều cao.
    
    Args:
        parent_container_grid (torch.Tensor): Tensor có shape (B, C, H_total, W_total)
        num_child_on_width_size (int): Số ảnh con theo chiều rộng (số cột).
        num_child_on_height_size (int): Số ảnh con theo chiều cao (số hàng).
        
    Returns:
        list: Danh sách các tensor con, mỗi tensor có shape (B, C, patch_H, patch_W)
    """
    B, C, H_total, W_total = parent_container_grid.shape

    # Kiểm tra chia hết để có thể tách đều
    if H_total % num_child_on_height_size != 0:
        raise ValueError(f"H_total ({H_total}) phải chia hết cho num_child_on_height_size ({num_child_on_height_size}).")
    if W_total % num_child_on_width_size != 0:
        raise ValueError(f"W_total ({W_total}) phải chia hết cho num_child_on_width_size ({num_child_on_width_size}).")

    patch_H = H_total // num_child_on_height_size
    patch_W = W_total // num_child_on_width_size
    patches = []

    # Duyệt qua từng hàng và cột của lưới
    for i in range(num_child_on_height_size):
        for j in range(num_child_on_width_size):
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
    Copyright Length: <số bit trong chuỗi copyright (64)>
        Copyright Before: <dãy 64 bit>
        Copyright After: <dãy 64 bit>
        Copyright Bit Error: <số bit khác nhau giữa copyright trước và sau chia cho 64>
        Copyright Wrong Position: <list các vị trí khác nhau>
    Metadata Length: <số bit trong metadata>
        Metadata Before: <dãy metadata, mỗi 64 bit cách nhau bởi dấu |>
        Metadata After: <dãy metadata, mỗi 64 bit cách nhau bởi dấu |>
        Metadata Bit Error: <số bit khác nhau giữa metadata trước và sau chia cho tổng số bit metadata>
        Metadata Wrong Position: <list các vị trí khác nhau>
        General Bit Length: <tổng số bit trong (copyright + metadata)>
        General Bit Before: <dãy (copyright + metadata) trước được chia theo 64-bit với dấu |>
        General Bit After: <dãy (copyright + metadata) sau được chia theo 64-bit với dấu |>
        General Bit Error: <số bit khác nhau trong toàn bộ dãy chia cho tổng số bit>
    General Wrong Position: <list các vị trí khác nhau trong toàn bộ dãy>
    
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
    # Format parent image id thành chuỗi 4 chữ số.
    image_id_str = f"{parent_image_id:04d}"
    
    # Kiểm tra độ dài của copyright và metadata.
    if len(copyright_before) != 64 or len(copyright_after) != 64:
        raise ValueError("Copyright trước và sau phải có đúng 64 bit.")
    
    if len(metadata_before) % 64 != 0 or len(metadata_after) % 64 != 0:
        raise ValueError("Metadata trước và sau phải là bội số của 64 bit.")
    
    # Tính toán lỗi bit cho copyright.
    copyright_diff_positions = [i for i in range(64) 
                                if copyright_before[i] != copyright_after[i]]
    copyright_diff_count = len(copyright_diff_positions)
    copyright_bit_error = copyright_diff_count / 64
    
    # Tính toán lỗi bit cho metadata.
    metadata_len = len(metadata_before)
    metadata_diff_positions = [i for i in range(metadata_len)
                               if metadata_before[i] != metadata_after[i]]
    metadata_diff_count = len(metadata_diff_positions)
    metadata_bit_error = metadata_diff_count / metadata_len
    
    # Tính toán lỗi bit cho toàn bộ dãy (copyright + metadata).
    combined_before = copyright_before + metadata_before
    combined_after = copyright_after + metadata_after
    combined_length = len(combined_before)  # = 64 + metadata_len
    general_diff_positions = [i for i in range(combined_length)
                              if combined_before[i] != combined_after[i]]
    combined_diff_count = len(general_diff_positions)
    general_bit_error = combined_diff_count / combined_length

    # Helper function: format bit string in blocks of 64 separated by " | "
    def format_in_blocks(bit_str, block_size=64):
        return " | ".join([bit_str[i:i+block_size] for i in range(0, len(bit_str), block_size)])
    
    # Format metadata and general bit strings.
    metadata_before_formatted = format_in_blocks(metadata_before)
    metadata_after_formatted  = format_in_blocks(metadata_after)
    general_before_formatted  = format_in_blocks(combined_before)
    general_after_formatted   = format_in_blocks(combined_after)
    
    # Ghi thông tin ra file theo định dạng mới.
    with open(out_file_path, 'a') as f:
        f.write(f"Image_ID: {image_id_str}\n")
        
        # Copyright block
        f.write("Copyright Length: 64\n")
        f.write(f"    Copyright Before: {copyright_before}\n")
        f.write(f"    Copyright After: {copyright_after}\n")
        f.write(f"    Copyright Bit Error: {copyright_bit_error}\n")
        f.write(f"    Copyright Wrong Position: {copyright_diff_positions}\n")
        
        # Metadata block
        f.write(f"Metadata Length: {metadata_len}\n")
        f.write(f"    Metadata Before: {metadata_before_formatted}\n")
        f.write(f"    Metadata After: {metadata_after_formatted}\n")
        f.write(f"    Metadata Bit Error: {metadata_bit_error}\n")
        f.write(f"    Metadata Wrong Position: {metadata_diff_positions}\n")
        
        # General block
        f.write(f"    General Bit Length: {combined_length}\n")
        f.write(f"    General Bit Before: {general_before_formatted}\n")
        f.write(f"    General Bit After: {general_after_formatted}\n")
        f.write(f"    General Bit Error: {general_bit_error}\n")
        f.write(f"General Wrong Position: {general_diff_positions}\n")
        f.write("---------------------\n")
    
    return general_bit_error
# ----- VN End -----
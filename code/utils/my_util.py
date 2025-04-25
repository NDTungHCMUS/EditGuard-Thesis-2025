# ----- VN Start -----
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
import os
import math
from .reed_solomons_16 import compute_parity_16, recover_original_16
from .reed_solomons_8 import compute_parity_8, recover_original_8
from .hamming_code_7_4 import compute_parity_hamming_74, recover_original_hamming_74
from .hamming_code_12_8 import compute_parity_hamming_12_8, recover_original_hamming_12_8
from .LDPC import ldpc_encode, ldpc_decode_bp
from .util import save_img, tensor2img, decoded_message_error_rate

def load_copyright_phash_metadata_from_files(
    file_path: str,
    number_of_64bits_blocks_copyright: int,
    number_of_64bits_blocks_phash: int,
    number_of_64bits_blocks_metadata: int
) -> list[dict]:
    """
    Đọc file có định dạng:
      0001
      <copyright block 1>
      ...
      <copyright block N>
      <phash block 1>
      ...
      <phash block M>
      <metadata block 1>
      ...
      <metadata block K>
      0002
      ...
    Trả về list các dict với keys: "copyright", "phash", "metadata".
    """
    results = []
    # Đọc vào và loại bỏ newline, giữ nguyên thứ tự
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    i = 0
    total_blocks = (
        number_of_64bits_blocks_copyright
        + number_of_64bits_blocks_phash
        + number_of_64bits_blocks_metadata
    )

    while i < len(lines):
        # 1) dòng image number
        if len(lines[i]) == 4 and lines[i].isdigit():
            i += 1
        else:
            raise ValueError(f"Expected image-number (4 digits) at line {i+1}, got: {lines[i]!r}")

        # 2) kiểm tra đủ số dòng còn lại để đọc hết 3 phần
        if i + total_blocks > len(lines):
            raise EOFError(f"Not enough lines for all blocks at image starting line {i}")

        # 3) đọc từng phần theo số block
        copyright_list = lines[i : i + number_of_64bits_blocks_copyright]
        i += number_of_64bits_blocks_copyright

        phash_list = lines[i : i + number_of_64bits_blocks_phash]
        i += number_of_64bits_blocks_phash

        metadata_list = lines[i : i + number_of_64bits_blocks_metadata]
        i += number_of_64bits_blocks_metadata

        results.append({
            "copyright": copyright_list,
            "phash":    phash_list,
            "metadata": metadata_list
        })

    return results


def compute_parity_from_list_copyright_phash_metadata(
    list_copyright_metadata,
    number_of_64bits_blocks_copyright,
    number_of_64bits_blocks_phash,
    number_of_64bits_blocks_metadata,
    type_correction_code=1,
    P=-1,
    H=-1
):
    """
    Với mỗi dict trong list_copyright_metadata (có keys "copyright", "phash", "metadata",
    mỗi key chứa một list các block 64‑bit):
      - Kiểm tra số block đúng với tham số number_of_64bits_...
      - Tính parity/ECC theo type_correction_code (và P/H nếu cần)
    Trả về list các dict cùng key nhưng chứa parity code.
    """
    def _encode(block: str) -> str:
        if   type_correction_code == 1:
            return compute_parity_16(block)
        elif type_correction_code == 2:
            return compute_parity_8(block)
        elif type_correction_code == 3:
            return compute_parity_hamming_74(block)
        elif type_correction_code == 4:
            return compute_parity_hamming_12_8(block)
        elif type_correction_code == 5:
            return ldpc_encode(block, P)
        else:
            return "0" * len(block)

    results = []
    for idx, item in enumerate(list_copyright_metadata):
        # Validate độ dài từng phần
        if len(item.get("copyright", [])) != number_of_64bits_blocks_copyright:
            raise ValueError(
                f"Item {idx}: expected {number_of_64bits_blocks_copyright} copyright blocks, "
                f"got {len(item.get('copyright', []))}"
            )
        if len(item.get("phash", [])) != number_of_64bits_blocks_phash:
            raise ValueError(
                f"Item {idx}: expected {number_of_64bits_blocks_phash} phash blocks, "
                f"got {len(item.get('phash', []))}"
            )
        if len(item.get("metadata", [])) != number_of_64bits_blocks_metadata:
            raise ValueError(
                f"Item {idx}: expected {number_of_64bits_blocks_metadata} metadata blocks, "
                f"got {len(item.get('metadata', []))}"
            )

        # Tính parity cho từng phần
        parity_copyright = [
            _encode(block) for block in item["copyright"]
        ]
        parity_phash    = [
            _encode(block) for block in item["phash"]
        ]
        parity_metadata = [
            _encode(block) for block in item["metadata"]
        ]

        results.append({
            "copyright": parity_copyright,
            "phash":    parity_phash,
            "metadata": parity_metadata
        })

    return results

# Convert index to position in the random walk sequence
## Return (index, type) where type = 0: copyright, 1: parity_copyright, 2: phash, 3: parity_phash, 4: metadata, 5: parity_metadata
def convert_index_to_position(index, random_walk_sequence, number_of_64bits_blocks_copyright, number_of_64bits_blocks_phash, number_of_64bits_blocks_metadata):
    for i in range(len(random_walk_sequence)):
        if (random_walk_sequence[i] == index):
            if (i < number_of_64bits_blocks_copyright):
                return i, 0
            elif (i < 2 * number_of_64bits_blocks_copyright):
                return i - number_of_64bits_blocks_copyright, 1
            elif (i < 2 * number_of_64bits_blocks_copyright + number_of_64bits_blocks_phash):
                return i - 2 * number_of_64bits_blocks_copyright, 2
            elif (i < 2 * number_of_64bits_blocks_copyright + 2 * number_of_64bits_blocks_phash):
                return i - 2 * number_of_64bits_blocks_copyright - number_of_64bits_blocks_phash, 3
            elif (i < 2 * number_of_64bits_blocks_copyright + 2 * number_of_64bits_blocks_phash + number_of_64bits_blocks_metadata):
                return i - 2 * number_of_64bits_blocks_copyright - 2 * number_of_64bits_blocks_phash, 4
            elif (i < 2 * number_of_64bits_blocks_copyright + 2 * number_of_64bits_blocks_phash + 2 * number_of_64bits_blocks_metadata):
                return i - 2 * number_of_64bits_blocks_copyright - 2 * number_of_64bits_blocks_phash - number_of_64bits_blocks_metadata, 5
    return -1, -1

## Explaination: Return message correspond to dictionary (copyright or metadata)
def compute_message(index, dict_copyright_phash_metadata, dict_parity_copyright_phash_metadata, random_walk_sequence, number_of_64bits_blocks_copyright, number_of_64bits_blocks_phash, number_of_64bits_blocks_metadata):
    n_c = number_of_64bits_blocks_copyright
    n_ph = number_of_64bits_blocks_phash
    n_m = number_of_64bits_blocks_metadata

    index_in_dict, group = convert_index_to_position(index, random_walk_sequence, n_c, n_ph, n_m)

    # Nếu không tìm thấy vị trí trong dict, trả về -1
    if (index_in_dict == -1 or group == -1):
        return -1
    
    keys = ["copyright", "phash", "metadata"]
    key = keys[group // 2]

    # chọn dict gốc hay dict_parity
    if group % 2 == 1:
        # parity
        source_dict = dict_parity_copyright_phash_metadata[index]
    else:
        source_dict = dict_copyright_phash_metadata[index]

    # trả về block
    try:
        return source_dict[key][index_in_dict]
    except (KeyError, IndexError) as e:
        return -1

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

## Explaination: Take copyright, phash, metadata (before and after) from lists (without correction)
def get_copyright_phash_metadata_from_list_without_correction(list_message, list_recmessage, random_walk_sequence, number_of_64bits_blocks_copyright, number_of_64bits_blocks_phash, number_of_64bits_blocks_metadata):
    copyright_before = ""
    copyright_after = ""
    phash_before = ""
    phash_after = ""
    metadata_before = ""
    metadata_after = ""
    for i in range(0, number_of_64bits_blocks_copyright):
        copyright_before = copyright_before + list_message[random_walk_sequence[i]]
        copyright_after = copyright_after + list_recmessage[random_walk_sequence[i]]
    for i in range(2 * number_of_64bits_blocks_copyright, 2 * number_of_64bits_blocks_copyright + number_of_64bits_blocks_phash):
        phash_before = phash_before + list_message[random_walk_sequence[i]]
        phash_after = phash_after + list_recmessage[random_walk_sequence[i]]
    for i in range(2 * number_of_64bits_blocks_copyright + 2 * number_of_64bits_blocks_phash, 2 * number_of_64bits_blocks_copyright + 2 * number_of_64bits_blocks_phash + number_of_64bits_blocks_metadata):
        metadata_before = metadata_before + list_message[random_walk_sequence[i]]
        metadata_after = metadata_after + list_recmessage[random_walk_sequence[i]]
    return copyright_before, copyright_after, phash_before, phash_after, metadata_before, metadata_after

## Explaination: Take copyright, metadata (before and after) from lists (with correction)
def get_copyright_phash_metadata_from_list_with_correction(
    list_message,
    list_recmessage,
    random_walk_sequence,
    num_copyright_blocks,
    num_phash_blocks,
    num_metadata_blocks,
    type_correction_code=1,
    H=None
):
    """
    Extracts and error-corrects copyright, pHash, and metadata blocks.

    Returns:
      (copyright_before, copyright_after,
       phash_before,     phash_after,
       metadata_before,  metadata_after,
       cnt_cannot_solve)
    """
    # Helper to dispatch to the right decoder
    def _recover(codeword):
        if type_correction_code == 1:
            return recover_original_16(codeword)
        elif type_correction_code == 2:
            return recover_original_8(codeword)
        elif type_correction_code == 3:
            return recover_original_hamming_74(codeword)
        elif type_correction_code == 4:
            return recover_original_hamming_12_8(codeword)
        elif type_correction_code == 5:
            return ldpc_decode_bp(codeword, H)
        else:
            return -1

    copyright_before = ""
    copyright_after  = ""
    phash_before     = ""
    phash_after      = ""
    metadata_before  = ""
    metadata_after   = ""
    cnt_cannot_solve = 0

    # ——— COPYRIGHT BLOCKS ———
    for i in range(num_copyright_blocks):
        data_idx   = random_walk_sequence[i]
        parity_idx = random_walk_sequence[num_copyright_blocks + i]
        # accumulate “before”
        copyright_before += list_message[data_idx]
        # build the codeword and attempt correction
        codeword = list_recmessage[data_idx] + list_recmessage[parity_idx]
        recovered = _recover(codeword)
        if recovered == -1:
            cnt_cannot_solve += 1
            # fall back to the (possibly corrupted) data portion
            copyright_after += list_recmessage[data_idx][:64]
        else:
            copyright_after += recovered[:64]

    # ——— pHASH BLOCKS ———
    phash_offset       = 2 * num_copyright_blocks
    phash_parity_start = phash_offset + num_phash_blocks
    for i in range(num_phash_blocks):
        data_idx   = random_walk_sequence[phash_offset + i]
        parity_idx = random_walk_sequence[phash_parity_start + i]
        phash_before += list_message[data_idx]
        codeword     = list_recmessage[data_idx] + list_recmessage[parity_idx]
        recovered    = _recover(codeword)
        if recovered == -1:
            cnt_cannot_solve += 1
            phash_after += list_recmessage[data_idx][:64]
        else:
            phash_after += recovered[:64]

    # ——— METADATA BLOCKS ———
    meta_offset       = 2 * num_copyright_blocks + 2 * num_phash_blocks
    meta_parity_start = meta_offset + num_metadata_blocks
    for i in range(num_metadata_blocks):
        data_idx   = random_walk_sequence[meta_offset + i]
        parity_idx = random_walk_sequence[meta_parity_start + i]
        metadata_before += list_message[data_idx]
        codeword       = list_recmessage[data_idx] + list_recmessage[parity_idx]
        recovered      = _recover(codeword)
        if recovered == -1:
            cnt_cannot_solve += 1
            metadata_after += list_recmessage[data_idx][:64]
        else:
            metadata_after += recovered[:64]

    return (
        copyright_before, copyright_after,
        phash_before,     phash_after,
        metadata_before,  metadata_after,
        cnt_cannot_solve
    )


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
def write_extracted_messages(
    parent_image_id,
    copyright_before,
    copyright_after,
    phash_before,
    phash_after,
    metadata_before,
    metadata_after,
    out_file_path
):
    """
    Ghi ra file thông tin trích xuất cho một ảnh với định dạng:
    
    Image_ID: <parent_image_id, định dạng 4 chữ số>
    Copyright Length: <số bit trong chuỗi copyright (bội số của 64)>
        Copyright Before: <dãy bit>
        Copyright After: <dãy bit>
        Copyright Bit Error: <số bit khác nhau giữa copyright trước và sau chia cho độ dài>
        Copyright Wrong Position: <list các vị trí khác nhau>
    pHash Length: <số bit trong chuỗi pHash (bội số của 64)>
        pHash Before: <dãy bit>
        pHash After: <dãy bit>
        pHash Bit Error: <số bit khác nhau giữa pHash trước và sau chia cho độ dài>
        pHash Wrong Position: <list các vị trí khác nhau>
    Metadata Length: <số bit trong metadata (bội số của 64)>
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
        copyright_before (str): Dãy bit copyright trước khi sửa (bội số của 64).
        copyright_after (str): Dãy bit copyright sau khi sửa (bội số của 64).
        phash_before (str): Dãy bit pHash trước khi sửa (bội số của 64).
        phash_after (str): Dãy bit pHash sau khi sửa (bội số của 64).
        metadata_before (str): Dãy metadata trước khi sửa (bội số của 64).
        metadata_after (str): Dãy metadata sau khi sửa (bội số của 64).
        out_file_path (str): Đường dẫn đến file output.
        
    Returns:
        float: Giá trị general_bit_error.
    """
    # Format parent image id thành chuỗi 4 chữ số.
    image_id_str = f"{parent_image_id:04d}"
    
    # Kiểm tra độ dài đầu vào đều là bội số của 64
    for name, before, after in [
        ("Copyright", copyright_before, copyright_after),
        ("pHash", phash_before, phash_after),
        ("Metadata", metadata_before, metadata_after)
    ]:
        if len(before) % 64 != 0 or len(after) % 64 != 0:
            raise ValueError(f"{name} trước và sau phải là bội số của 64 bit.")
    
    # Tính toán lỗi bit cho từng phần
    def bit_diff_stats(before, after):
        length = len(before)
        positions = [i for i in range(length) if before[i] != after[i]]
        error_rate = len(positions) / length
        return length, positions, error_rate
    
    cp_len,   cp_pos,   cp_err   = bit_diff_stats(copyright_before, copyright_after)
    ph_len,   ph_pos,   ph_err   = bit_diff_stats(phash_before, phash_after)
    md_len,   md_pos,   md_err   = bit_diff_stats(metadata_before, metadata_after)
    
    # Tính toán lỗi bit cho toàn bộ dãy (copyright + metadata)
    combined_before = copyright_before + phash_before + metadata_before
    combined_after  = copyright_after  + phash_after + metadata_after
    gen_len, gen_pos, gen_err = bit_diff_stats(combined_before, combined_after)

    # Helper: format bit string in blocks of 64 separated by " | "
    def format_in_blocks(bit_str, block_size=64):
        return " | ".join(
            bit_str[i:i+block_size] for i in range(0, len(bit_str), block_size)
        )
    
    copyright_before_fmt = format_in_blocks(copyright_before)
    copyright_after_fmt  = format_in_blocks(copyright_after)
    phash_before_fmt    = format_in_blocks(phash_before)
    phash_after_fmt     = format_in_blocks(phash_after)
    metadata_before_fmt = format_in_blocks(metadata_before)
    metadata_after_fmt  = format_in_blocks(metadata_after)
    general_before_fmt  = format_in_blocks(combined_before)
    general_after_fmt   = format_in_blocks(combined_after)
    
    # Ghi thông tin ra file
    with open(out_file_path, 'a') as f:
        f.write(f"Image_ID: {image_id_str}\n")
        
        # Copyright block
        f.write(f"Copyright Length: {cp_len}\n")
        f.write(f"    Copyright Before: {copyright_before_fmt}\n")
        f.write(f"    Copyright After: {copyright_after_fmt}\n")
        f.write(f"    Copyright Bit Error: {cp_err}\n")
        f.write(f"    Copyright Wrong Position: {cp_pos}\n")
        
        # pHash block
        f.write(f"pHash Length: {ph_len}\n")
        f.write(f"    pHash Before: {phash_before_fmt}\n")
        f.write(f"    pHash After: {phash_after_fmt}\n")
        f.write(f"    pHash Bit Error: {ph_err}\n")
        f.write(f"    pHash Wrong Position: {ph_pos}\n")
        
        # Metadata block
        f.write(f"Metadata Length: {md_len}\n")
        f.write(f"    Metadata Before: {metadata_before_fmt}\n")
        f.write(f"    Metadata After: {metadata_after_fmt}\n")
        f.write(f"    Metadata Bit Error: {md_err}\n")
        f.write(f"    Metadata Wrong Position: {md_pos}\n")
        
        # General block
        f.write(f"General Bit Length: {gen_len}\n")
        f.write(f"    General Bit Before: {general_before_fmt}\n")
        f.write(f"    General Bit After: {general_after_fmt}\n")
        f.write(f"    General Bit Error: {gen_err}\n")
        f.write(f"General Wrong Position: {gen_pos}\n")
        f.write("---------------------\n")
    
    return gen_err
# ----- VN End -----
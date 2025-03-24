import torch
import torchvision.transforms.functional as TF
from PIL import Image
import os
import numpy as np
import math
from .util import save_img, tensor2img, decoded_message_error_rate

def split_and_save_image_torch(image_path, output_folder="images", grid_size=6):
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

def split_all_images(input_folder="A", output_folder="B"):
    """
    Duyệt qua tất cả các ảnh trong thư mục input_folder, chia nhỏ và lưu vào output_folder.

    Args:
        input_folder (str): Thư mục chứa ảnh gốc.
        output_folder (str): Thư mục để lưu ảnh đã cắt.
    """
    os.makedirs(output_folder, exist_ok=True)  # Đảm bảo thư mục đầu ra tồn tại

    for filename in os.listdir(input_folder):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(input_folder, filename)
            img_name, _ = os.path.splitext(filename)  # Lấy tên file không có đuôi
            save_dir = os.path.join(output_folder, img_name)  # Tạo thư mục riêng cho ảnh
            os.makedirs(save_dir, exist_ok=True)

            split_and_save_image_torch(img_path, save_dir)

# split_and_save_image_torch("input.jpg") 

# Combine tất cả ảnh trong thư mục input thành 1 ảnh lớn lưu trong thư mục output
def combine_images_from_folder(input_folder, output_folder, num_images=36):
    """
    Gom num_images ảnh trong folder thành một ảnh lớn theo dạng lưới √num_images x √num_images.
    Mặc định num_images = 36 (6x6). Có thể thay đổi thành 4 (2x2) hoặc 9 (3x3).
    
    Args:
        input_folder (str): Đường dẫn chứa các ảnh nhỏ (ví dụ: a/dataset/0001).
        output_folder (str): Đường dẫn chứa ảnh tổng hợp (ví dụ: b/dataset).
        num_images (int): Số lượng ảnh muốn ghép (nên là số chính phương: 4, 9, 16, 25, 36,...).
    """
    # Kiểm tra num_images có phải số chính phương hay không
    root = int(math.sqrt(num_images))
    if root * root != num_images:
        raise ValueError(f"num_images = {num_images} không phải số chính phương (4, 9, 16, 25, 36, ...)")

    # Tạo thư mục đầu ra nếu chưa tồn tại
    os.makedirs(output_folder, exist_ok=True)

    # Lấy danh sách file .png trong thư mục
    all_files = sorted([
        f for f in os.listdir(input_folder)
        if f.lower().endswith('.png')
    ])

    # Kiểm tra xem đủ ảnh hay không
    if len(all_files) < num_images:
        raise ValueError(
            f"Số ảnh trong thư mục {input_folder} không đủ (tìm thấy {len(all_files)}, yêu cầu {num_images})."
        )

    # Chỉ lấy đúng num_images file đầu tiên (hoặc theo nhu cầu)
    selected_files = all_files[:num_images]

    # Đọc ảnh từ danh sách file
    patches = []
    for filename in selected_files:
        patch_path = os.path.join(input_folder, filename)
        patch = Image.open(patch_path)
        patches.append(patch)

    # Giả sử tất cả ảnh có cùng kích thước
    patch_width, patch_height = patches[0].size

    # Tạo ảnh mới kích thước: (root * patch_width) x (root * patch_height)
    combined_width = patch_width * root
    combined_height = patch_height * root
    combined_image = Image.new("RGB", (combined_width, combined_height))

    # Dán từng ảnh vào vị trí tương ứng
    for idx, patch in enumerate(patches):
        row = idx // root
        col = idx % root
        combined_image.paste(patch, (col * patch_width, row * patch_height))

    # Lấy tên folder cuối của input_folder làm tên file đầu ra
    folder_name = os.path.basename(os.path.normpath(input_folder))
    output_file_path = os.path.join(output_folder, f"{folder_name}.png")

    # Lưu ảnh tổng hợp
    combined_image.save(output_file_path)
    print(f"Đã lưu ảnh tổng hợp: {output_file_path}")

# Duyệt qua tât cả các folder con trong input_dataset_folder và gom ảnh từng folder con thành 1 ảnh lớn
def combine_all_images(input_dataset_folder, output_folder):
    """
    Lặp qua toàn bộ folder con trong input_dataset_folder (ví dụ: a/dataset/000x)
    và thực hiện ghép 36 ảnh trong mỗi folder con thành một ảnh lớn.
    
    Args:
        input_dataset_folder (str): Đường dẫn chứa các folder con (ví dụ: a/dataset).
        output_folder (str): Đường dẫn chứa ảnh tổng hợp (ví dụ: b/dataset).
    """
    # Lấy danh sách tất cả các folder con trong input_dataset_folder
    subfolders = [entry.path for entry in os.scandir(input_dataset_folder) if entry.is_dir()]
    
    if not subfolders:
        print("Không tìm thấy folder con nào trong", input_dataset_folder)
        return
    
    for subfolder in subfolders:
        try:
            combine_images_from_folder(subfolder, output_folder)
        except Exception as e:
            print(f"Lỗi khi xử lý folder {subfolder}: {e}")

# Tạo một tensor chứa n^2 ảnh từ n^2 tensor nhỏ (4 chiều)
def combine_torch_tensors_4d(list_container, num_images=None):
    """
    Kết hợp danh sách các tensor 4D (B, C, H, W) thành một tensor 4D (B, C, newH, newW),
    sắp xếp theo dạng lưới √num_images x √num_images.
    
    Args:
        list_container (List[torch.Tensor]): Danh sách các tensor 4D, mỗi tensor shape (B, C, H, W).
        num_images (int, optional): Số lượng tensor sẽ được ghép. Mặc định = len(list_container).
            Yêu cầu: num_images phải là số chính phương (4, 9, 16, 25, 36, ...).
    
    Returns:
        torch.Tensor: Tensor 4D có shape (B, C, root*H, root*W), trong đó root = √num_images.
    """
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

# Chia một tensor lớn thành list n^2 tensor nhỏ (4 chiều)
def split_torch_tensors_4d(parent_container_grid, grid_size=6):
    """
    Chia một tensor 4D (B, C, H_total, W_total) thành grid_size x grid_size patch nhỏ theo lưới.
    
    Output:
      - Một list gồm grid_size * grid_size tensor, mỗi tensor có shape (B, C, patch_H, patch_W),
        với patch_H = H_total // grid_size và patch_W = W_total // grid_size.
    
    Yêu cầu:
      - H_total và W_total phải chia hết cho grid_size.
    
    Ví dụ:
      - Nếu grid_size=6 (mặc định) => chia thành 36 patch (6x6).
      - Nếu grid_size=3 => chia thành 9 patch (3x3).
      - Nếu grid_size=2 => chia thành 4 patch (2x2).
    """
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

# combine_all_images("E:/Year_4/Thesis/EditGuard-Split-Image/dataset/valAGE-Set-5-eles-split-ori", "E:/Year_4/Thesis/EditGuard-Split-Image/dataset/valAGE-Set-5-eles-merge")  # Gom ảnh từ thư mục images và lưu ra output.jpg

def tensor_to_pil(tensor):
    """
    Chuyển một tensor có ít nhất 3 chiều thành ảnh PIL.Image bằng cách chỉ lấy 3 chiều cuối.
    
    Ví dụ:
    - Nếu tensor có shape (1, 3, 512, 512) → chuyển thành (3, 512, 512)
    - Nếu tensor có shape (B, N, 3, H, W) → chuyển thành (3, H, W) (bỏ qua các chiều trước)
    """
    # Nếu tensor đang nằm trên GPU, chuyển về CPU
    if tensor.device != torch.device('cpu'):
        tensor = tensor.cpu()
    
    # Nếu tensor có nhiều hơn 3 chiều, chỉ lấy 3 chiều cuối
    if tensor.dim() > 3:
        # Sử dụng view để lấy shape của 3 chiều cuối
        tensor = tensor.contiguous().view(*tensor.shape[-3:])
    
    # Chuyển tensor thành PIL Image (giả sử giá trị tensor nằm trong khoảng [0, 1] hoặc [0, 255])
    return TF.to_pil_image(tensor)

# Cho 1 list tensor, id của ảnh cha, lưu ảnh vào thư mục output_dir/parent_image_id/child_image_id.png
def save_tensor_images(list_container, parent_image_id, output_dir='a', out_type=np.uint8, min_max=(0,1)):
    """
    Lưu danh sách các tensor thành ảnh bằng cách sử dụng hàm save_img.
    Ảnh sẽ được lưu vào thư mục: output_dir/parent_image_id.zfill(4)
    Tên file của ảnh sẽ có định dạng: i.zfill(4) + '.png' (ví dụ: 0001.png, 0002.png, ...)

    Args:
        list_container (list): Danh sách các tensor.
        parent_image_id (str hoặc int): ID của ảnh cha để tạo folder.
        output_dir (str): Thư mục gốc để lưu ảnh (mặc định 'a').
        out_type: Kiểu dữ liệu đầu ra của ảnh (mặc định np.uint8).
        min_max (tuple): Khoảng giá trị để chuẩn hóa tensor (mặc định (0,1)).
    """
    folder_name = str(parent_image_id + 1).zfill(4)
    output_folder = os.path.join(output_dir, folder_name)
    os.makedirs(output_folder, exist_ok=True)

    for i, tensor in enumerate(list_container):
        # Chuyển tensor thành mảng numpy đại diện cho ảnh
        img_np = tensor2img(tensor, out_type=out_type, min_max=min_max)
        
        # Tạo tên file theo định dạng: 0001.png, 0002.png, ...
        file_name = f"{i}.png"
        file_path = os.path.join(output_folder, file_name)
        
        # Lưu ảnh sử dụng hàm save_img
        save_img(img_np, file_path)


def write_extracted_messages(parent_image_id, list_message, list_recmessage, out_file_path):
    """
    Append thông tin của các ảnh con vào file text theo định dạng:
      Line 1: parent_image_id,child_image_id
      Line 2: message (dưới dạng tensor)
      Line 3: recmessage (dưới dạng tensor)
      Line 4: error_rate của cặp
      Line 5: dòng trống phân cách

    Sau đó, ghi và in thêm một dòng cho trung bình error rate của tất cả các ảnh con.

    Nếu message và recmessage là tensor, giữ nguyên tensor và tính error rate bằng
    hàm decoded_message_error_rate (hoặc decoded_message_error_rate_batch) đã định nghĩa sẵn.

    Args:
        parent_image_id (int hoặc str): ID của ảnh cha.
        list_message (list): Danh sách các message từ 36 ảnh con.
        list_recmessage (list): Danh sách các recmessage từ 36 ảnh con.
        out_file_path (str): Đường dẫn file text sẽ được lưu.
    """
    import os
    import torch

    error_rates = []  # Lưu error_rate cho từng cặp message/recmessage

    # Kiểm tra file có tồn tại không, để quyết định ghi header hay append
    file_exists = os.path.exists(out_file_path)
    with open(out_file_path, 'a', encoding='utf-8') as f:
        if not file_exists:
            f.write("Extracted Messages:\n\n")
        
        for child_image_id, (message, recmessage) in enumerate(zip(list_message, list_recmessage)):
            # Tính error rate cho cặp message, recmessage
            try:
                # Gọi hàm decoded_message_error_rate (giữ nguyên không sửa)
                error_rate = decoded_message_error_rate(message, recmessage)
            except Exception as e:
                # Nếu có lỗi (ví dụ do việc convert tensor với nhiều phần tử), sử dụng fallback:
                message_flat = message.view(message.shape[0], -1).squeeze()
                recmessage_flat = recmessage.view(recmessage.shape[0], -1).squeeze()
                error_count = sum([int(x.item()) for x in (message_flat.gt(0) != recmessage_flat.gt(0))])
                length = message_flat.numel()
                error_rate = error_count / length

            # Nếu error_rate là tensor, chuyển về float
            if isinstance(error_rate, torch.Tensor):
                error_rate = error_rate.item()
            
            error_rates.append(error_rate)
            
            # Giữ nguyên tensor bằng string representation
            message_str = str(message)
            recmessage_str = str(recmessage)
            
            # Định dạng parent_image_id và child_image_id thành 4 chữ số
            parent_str = str(parent_image_id).zfill(4)
            child_str = str(child_image_id).zfill(4)
            
            # Ghi vào file với định dạng theo yêu cầu:
            # Line 1: parent_image_id,child_image_id
            f.write(f"{parent_str},{child_str}\n")
            # Line 2: message
            f.write(f"{message_str}\n")
            # Line 3: recmessage
            f.write(f"{recmessage_str}\n")
            # Line 4: error_rate
            f.write(f"{error_rate}\n")
            # Line 5: dòng trống phân cách
            f.write("\n")
            
            # In ra theo định dạng: message -> xuống hàng recmessage -> xuống hàng error
            print(f"{message_str}\n{recmessage_str}\n{error_rate}\n")
        
        # Tính trung bình error rate nếu có giá trị tính được
        if error_rates:
            avg_error = sum(error_rates) / len(error_rates)
        else:
            avg_error = "N/A"
        
        # Ghi và in ra trung bình error rate
        f.write("Average error rate:\n")
        f.write(f"{avg_error}\n")
        print("Average error rate:")
        print(avg_error)
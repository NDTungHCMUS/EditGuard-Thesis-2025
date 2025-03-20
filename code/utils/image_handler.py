import torch
import torchvision.transforms.functional as TF
from PIL import Image
import os
import numpy as np
from .util import save_img, tensor2img

def split_and_save_image_torch(image_path, output_folder="images"):
    """
    Chia ảnh thành 36 phần bằng nhau và lưu vào thư mục images.

    Args:
        image_path (str): Đường dẫn đến ảnh.
        output_folder (str): Thư mục để lưu ảnh.
    """
    # Đọc ảnh với PIL
    img = Image.open(image_path).convert("RGB")
    img_tensor = TF.to_tensor(img)  # Chuyển thành tensor (C, H, W)

    C, H, W = img_tensor.shape
    h, w = H // 6, W // 6  # Kích thước mỗi patch
    
    os.makedirs(output_folder, exist_ok=True)  # Tạo thư mục nếu chưa có

    count = 0
    for i in range(6):
        for j in range(6):
            patch = img_tensor[:, i*h:(i+1)*h, j*w:(j+1)*w]  # Cắt từng phần
            patch_img = TF.to_pil_image(patch)  # Chuyển lại thành ảnh PIL
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
def combine_images_from_folder(input_folder, output_folder):
    """
    Gom 36 ảnh trong folder thành một ảnh lớn theo dạng lưới 6x6.
    Các ảnh nhỏ được đặt tên: 0.png, 1.png, ..., 35.png.

    Args:
        input_folder (str): Đường dẫn chứa các ảnh nhỏ (ví dụ: a/dataset/0001).
        output_folder (str): Đường dẫn chứa ảnh tổng hợp (ví dụ: b/dataset).
    """
    # Tạo folder chứa ảnh tổng hợp nếu chưa tồn tại
    os.makedirs(output_folder, exist_ok=True)
    
    # Load 36 ảnh theo thứ tự từ 0.png đến 35.png
    patches = []
    for i in range(36):
        patch_path = os.path.join(input_folder, f"{i}.png")
        if not os.path.exists(patch_path):
            raise FileNotFoundError(f"File {patch_path} không tồn tại!")
        patch = Image.open(patch_path)
        patches.append(patch)
    
    # Giả sử tất cả các ảnh đều có cùng kích thước
    patch_width, patch_height = patches[0].size
    
    # Tạo ảnh mới với kích thước: chiều rộng = 6 * patch_width, chiều cao = 6 * patch_height
    combined_width = patch_width * 6
    combined_height = patch_height * 6
    combined_image = Image.new("RGB", (combined_width, combined_height))
    
    # Dán từng ảnh vào vị trí tương ứng trong ảnh tổng hợp
    for idx, patch in enumerate(patches):
        row = idx // 6  # hàng (0 đến 5)
        col = idx % 6   # cột (0 đến 5)
        combined_image.paste(patch, (col * patch_width, row * patch_height))
    
    # Lấy tên folder cuối cùng của input_folder làm tên file đầu ra
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

# Tạo một tensor chứa 36 ảnh từ 36 tensor nhỏ (4 chiều)
def combine_torch_tensors_4d(list_container):
    """
    Kết hợp danh sách 36 tensor, mỗi tensor có dạng (B, C, H, W), thành một tensor tổng hợp dạng lưới 6x6.
    Output: Tensor có dạng (B, C, 6*H, 6*W).
    
    Yêu cầu: Tất cả các tensor trong list_container phải có cùng batch size, số kênh (C), chiều cao (H) và chiều rộng (W).
    """
    if len(list_container) != 36:
        raise ValueError("Danh sách phải chứa đúng 36 tensor.")
    
    # Lấy thông tin từ tensor đầu tiên
    B, C, H, W = list_container[0].shape
    # Kiểm tra rằng tất cả các tensor đều có cùng shape
    for idx, tensor in enumerate(list_container):
        if tensor.shape != (B, C, H, W):
            raise ValueError(f"Tensor tại index {idx} có shape {tensor.shape} không khớp với tensor đầu tiên ({B, C, H, W}).")
    
    grid_list = []
    # Với mỗi mẫu trong batch, tạo ảnh lưới 6x6
    for b in range(B):
        # Lấy ra ảnh con tương ứng từ mỗi tensor, mỗi ảnh có shape (C, H, W)
        images = [tensor[b] for tensor in list_container]
        rows = []
        for i in range(6):
            # Ghép 6 ảnh theo chiều width (axis=2) để tạo thành 1 hàng
            row = torch.cat(images[i*6:(i+1)*6], dim=2)
            rows.append(row)
        # Ghép 6 hàng theo chiều height (axis=1) để tạo ra ảnh lưới cho mẫu thứ b
        grid_image = torch.cat(rows, dim=1)
        # Thêm batch dimension cho ảnh lưới
        grid_list.append(grid_image.unsqueeze(0))
    
    # Kết hợp lại tất cả các mẫu trong batch thành tensor 4D
    result = torch.cat(grid_list, dim=0)  # Shape: (B, C, 6*H, 6*W)
    return result

# Chia một tensor lớn thành list 36 tensor nhỏ (4 chiều)
def split_torch_tensors_4d(parent_container_grid):
    """
    Chia một tensor 4D (B, C, H_total, W_total) thành 36 patch nhỏ theo lưới 6x6.
    
    Output:
      - Một list gồm 36 tensor, mỗi tensor có shape (B, C, patch_H, patch_W), 
        với patch_H = H_total//6 và patch_W = W_total//6.
    
    Yêu cầu:
      - H_total và W_total phải chia hết cho 6.
    """
    B, C, H_total, W_total = parent_container_grid.shape
    patch_H = H_total // 6
    patch_W = W_total // 6
    patches = []
    
    for i in range(6):      # Duyệt qua 6 hàng
        for j in range(6):  # Duyệt qua 6 cột
            patch = parent_container_grid[:, :, 
                     i * patch_H:(i + 1) * patch_H, 
                     j * patch_W:(j + 1) * patch_W]
            patches.append(patch)
    
    return patches

combine_all_images("E:/Year_4/Thesis/EditGuard-Split-Image/dataset/valAGE-Set-5-eles-split-ori", "E:/Year_4/Thesis/EditGuard-Split-Image/dataset/valAGE-Set-5-eles-merge")  # Gom ảnh từ thư mục images và lưu ra output.jpg

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
    Ghi thông tin các hàng vào file text với định dạng:
    parent_image_id, child_image_id, message, recmessage
    
    Args:
        parent_image_id (int hoặc str): ID của ảnh cha.
        list_message (list): Danh sách các message từ 36 ảnh con.
        list_recmessage (list): Danh sách các recmessage từ 36 ảnh con.
        out_file_path (str): Đường dẫn file text sẽ được lưu.
    """
    with open(out_file_path, 'w', encoding='utf-8') as f:
        # Ghi header nếu cần
        f.write("parent_image_id,child_image_id,message,recmessage\n")
        # Giả sử list_message và list_recmessage có độ dài 36
        for child_image_id, (message, recmessage) in enumerate(zip(list_message, list_recmessage)):
            # Định dạng parent_image_id và child_image_id thành 4 chữ số
            parent_str = str(parent_image_id).zfill(4)
            child_str = str(child_image_id).zfill(4)
            # Ghi ra một dòng
            f.write(f"{parent_str},{child_str},{message},{recmessage}\n")
# ----- VN Start -----
import random

def random_walk_unique(n=21, min_val=1, max_val=31):
    """
    Sinh n số nguyên khác nhau theo thuật toán random walk trong khoảng [min_val, max_val].

    Args:
        n (int): Số lượng số cần sinh (mặc định 21).
        min_val (int): Giá trị nhỏ nhất (mặc định 1).
        max_val (int): Giá trị lớn nhất (mặc định 31).

    Returns:
        list: Danh sách các số nguyên.
    """
    # Chọn số khởi tạo ngẫu nhiên trong khoảng [min_val, max_val]
    start = random.randint(min_val, max_val)
    numbers = [start]  # Lưu giữ thứ tự sinh ra bằng list

    # Các bước nhảy có thể có (để tăng tính ngẫu nhiên)
    possible_steps = [-3, -2, -1, 1, 2, 3]

    while len(numbers) < n:
        current = random.choice(numbers)  # Chọn ngẫu nhiên một số đã có
        step = random.choice(possible_steps)  # Chọn bước nhảy ngẫu nhiên
        next_num = current + step
        # Nếu số mới nằm trong khoảng và chưa có trong list, thêm vào
        if min_val <= next_num <= max_val and next_num not in numbers:
            numbers.append(next_num)
    numbers.insert(0, 0)
    return numbers  # Trả về danh sách số nguyên thay vì chuỗi nhị phân
# ----- VN End -----
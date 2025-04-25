# ----- VN Start -----
import random


def random_walk_unique(number_of_64bits_blocks_input, num_child_images, input_number, seed=None):
    """
    Sinh đường đi ngẫu nhiên độc nhất (random walk) gồm 2 * number_of_64bits_blocks_input chỉ mục,
    mỗi chỉ mục trong khoảng [0, num_child_images-1], theo thuật toán:
      1) Khởi tạo PRNG với hạt seed = input_number (hoặc seed nếu được cung cấp).
      2) Lặp total = 2 * number_of_64bits_blocks_input lần:
         a) Lấy ngẫu nhiên idx = rng.randrange(num_child_images).
         b) Nếu idx đã có trong kết quả, tăng idx = (idx + 1) % num_child_images cho đến khi tìm được giá trị chưa có.
         c) Đưa idx vào kết quả.
    Đảm bảo mỗi giá trị chỉ xuất hiện một lần.

    Args:
        number_of_64bits_blocks_input (int): Số khối dữ liệu 64-bit ban đầu.
        num_child_images (int): Tổng số chỉ mục khả dụng (0 đến num_child_images-1).
        input_number (int): Giá trị đầu vào dùng làm hạt cho thuật toán.
        seed (int, optional): Hạt bổ sung nếu muốn ghi đè input_number.

    Returns:
        List[int]: Danh sách các chỉ mục duy nhất độ dài 2*number_of_64bits_blocks_input.

    Raises:
        ValueError: Nếu 2*number_of_64bits_blocks_input > num_child_images.
    """
    total = number_of_64bits_blocks_input * 2
    if total > num_child_images:
        raise ValueError(
            f"Cannot select {total} unique indices from {num_child_images} elements"
        )

    rng_seed = seed if seed is not None else input_number
    rng = random.Random(rng_seed)
    result = []

    for _ in range(total):
        idx = rng.randrange(num_child_images)
        # nếu idx trùng, plus 1 modulo cho đến khi tìm được cái mới
        while idx in result:
            idx = (idx + 1) % num_child_images
        result.append(idx)

    return result


# ----- VN End -----


if __name__ == "__main__":
    # Ví dụ test hàm random_walk_unique
    tests = [
        (2, 10, 123),
        (3, 10, 123),
        (4, 20, 999),
        (5, 12, 42),
    ]

    for blocks, total_images, seed in tests:
        print(f"Test: blocks={blocks}, total_images={total_images}, seed={seed}")
        path = random_walk_unique(blocks, total_images, seed)
        print("Result:", path)
        # Kiểm tra độ dài
        assert len(path) == 2 * blocks, f"Expected length {2*blocks}, got {len(path)}"
        # Kiểm tra các phần tử duy nhất và trong khoảng hợp lệ
        assert len(set(path)) == len(path), "Elements are not unique"
        assert all(0 <= x < total_images for x in path), "Index out of range"
        print("-> Passed\n")

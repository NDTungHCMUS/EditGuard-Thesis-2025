from .reed_solomons import compute_parity

# Create list data for all blocks
def create_list_data(random_walk_sequence, copyright, metadata):
    result = {}
    random_walk_block_1 = ''.join(format(num, '05b') for num in random_walk_sequence[1:13]) + "0000"
    random_walk_block_2 = ''.join(format(num, '05b') for num in random_walk_sequence[13:]) + '0' * 19
    result[random_walk_sequence[0]] = random_walk_block_1
    result[random_walk_sequence[1]] = compute_parity(random_walk_block_1)
    result[random_walk_sequence[2]] = random_walk_block_2
    result[random_walk_sequence[3]] = compute_parity(random_walk_block_2)
    result[random_walk_sequence[4]] = copyright
    result[random_walk_sequence[5]] = compute_parity(copyright)
    index = 6
    for i in range(8):
        metadata_chunk = metadata[i*64:(i+1)*64]  # Lấy 64 bit mỗi lần
        result[random_walk_sequence[index + i]] = metadata_chunk
        result[random_walk_sequence[index + i + 1]] = compute_parity(metadata_chunk)
        index += 1
    return result
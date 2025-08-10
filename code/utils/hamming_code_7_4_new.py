from typing import List, Tuple

# ---------------- Hamming(7,4) helpers ----------------

def encode_hamming74(data4: str) -> str:
    """
    data4: 'abcd' (4 bit '0'/'1')
    Trả về codeword 7 bit theo bố trí: [p1, p2, d1, p3, d2, d3, d4]
    """
    if len(data4) != 4 or any(c not in '01' for c in data4):
        raise ValueError("encode_hamming74: input phải là chuỗi 4 bit.")
    d1, d2, d3, d4 = map(int, data4)
    p1 = (d1 + d2 + d4) % 2          # kiểm tra các vị trí 1,3,5,7
    p2 = (d1 + d3 + d4) % 2          # kiểm tra các vị trí 2,3,6,7
    p3 = (d2 + d3 + d4) % 2          # kiểm tra các vị trí 4,5,6,7
    return f"{p1}{p2}{d1}{p3}{d2}{d3}{d4}"

def decode_hamming74(code7: str) -> Tuple[str, int]:
    """
    code7: 7 bit theo bố trí [p1, p2, d1, p3, d2, d3, d4]
    Trả về (data4_corrected, error_pos)
      - data4_corrected: 4 bit dữ liệu đã sửa
      - error_pos: vị trí bit lỗi (1..7) đã được sửa; 0 nếu không lỗi
    """
    if len(code7) != 7 or any(c not in '01' for c in code7):
        raise ValueError("decode_hamming74: input phải là chuỗi 7 bit.")
    bits = list(map(int, code7))
    #   idx: 1  2  3  4  5  6  7
    # bits: p1 p2 d1 p3 d2 d3 d4
    p1, p2, d1, p3, d2, d3, d4 = bits

    s1 = (p1 ^ d1 ^ d2 ^ d4)  # check parity set 1,3,5,7
    s2 = (p2 ^ d1 ^ d3 ^ d4)  # check parity set 2,3,6,7
    s3 = (p3 ^ d2 ^ d3 ^ d4)  # check parity set 4,5,6,7

    error_pos = s1 + (s2 << 1) + (s3 << 2)  # 1..7, 0 nếu không lỗi

    if error_pos != 0:
        # Sửa bit tại vị trí error_pos (1-indexed)
        idx0 = error_pos - 1
        bits[idx0] ^= 1
        p1, p2, d1, p3, d2, d3, d4 = bits

    data4 = f"{d1}{d2}{d3}{d4}"
    return data4, error_pos

# ---------------- 30-bit parity & recovery ----------------

def parity_30_from_30(data30: str) -> str:
    """
    Từ 30 bit dữ liệu → 30 bit parity:
      - Chia 30 bit thành 8 nibble 4-bit (nibble cuối pad '0' nếu thiếu)
      - Mỗi nibble tạo 3 parity (p1,p2,p3) từ Hamming(7,4)
      - Ghép 24 parity lại, rồi pad '0' → 30 bit
    """
    if len(data30) != 30 or any(c not in '01' for c in data30):
        raise ValueError("parity_30_from_30: input phải là chuỗi 30 bit.")

    # Tạo 8 nibble (ceil(30/4) = 8)
    nibbles: List[str] = []
    for i in range(0, 30, 4):
        chunk = data30[i:i+4]
        if len(chunk) < 4:
            chunk = chunk + '0' * (4 - len(chunk))  # pad cuối cho đủ 4
        nibbles.append(chunk)
    # nibbles có độ dài 8

    parity_bits = []
    for nib in nibbles:
        cw = encode_hamming74(nib)
        # Lấy p1, p2, p3 (vị trí 1,2,4) tức index 0,1,3
        parity_bits.extend([cw[0], cw[1], cw[3]])

    # 24 bit parity → pad lên 30
    parity_str = ''.join(parity_bits)
    if len(parity_str) < 30:
        parity_str += '0' * (30 - len(parity_str))
    return parity_str

def recover_30_from_codeword60(codeword60: str) -> str:
    """
    Từ codeword 60 bit (30 data + 30 parity) → khôi phục 30 bit dữ liệu.
    Cho phép sửa tối đa 1 lỗi/nhóm 7-bit (mỗi nibble độc lập).
    6 bit parity pad ở cuối bị bỏ qua khi giải mã.
    """
    if len(codeword60) != 60 or any(c not in '01' for c in codeword60):
        raise ValueError("recover_30_from_codeword60: input phải là chuỗi 60 bit.")

    data_part   = codeword60[:30]
    parity_part = codeword60[30:]

    # Chia data thành 8 nibble (nibble cuối có pad 0 logic)
    data_nibbles: List[str] = []
    for i in range(0, 30, 4):
        nib = data_part[i:i+4]
        if len(nib) < 4:
            nib = nib + '0' * (4 - len(nib))
        data_nibbles.append(nib)

    # Lấy 24 parity thực (bỏ 6 pad ở cuối)
    parity_real = parity_part[:24]
    if len(parity_real) != 24:
        raise RuntimeError("Parity section thiếu dữ liệu thực (phải >= 24 bit).")

    # Cắt parity thành 8 nhóm (p1,p2,p3)
    parity_triplets = [parity_real[i:i+3] for i in range(0, 24, 3)]

    # Với mỗi nibble: ghép thành codeword 7-bit theo bố trí [p1,p2,d1,p3,d2,d3,d4]
    recovered_bits = []
    for nib, ppp in zip(data_nibbles, parity_triplets):
        d1, d2, d3, d4 = nib
        p1, p2, p3 = ppp
        code7 = f"{p1}{p2}{d1}{p3}{d2}{d3}{d4}"
        data4_corr, _ = decode_hamming74(code7)
        recovered_bits.append(data4_corr)

    # Ghép lại và cắt về đúng 30 bit (bỏ 2 bit pad của nibble cuối)
    recovered = ''.join(recovered_bits)[:30]
    return recovered

# ---------------- Tiện ích: tạo codeword từ 30-bit ----------------

def build_codeword60_from_data30(data30: str) -> str:
    """
    Tạo codeword 60 bit từ dữ liệu 30 bit:
      codeword = data30 (30 bit) + parity_30_from_30(data30) (30 bit)
    """
    return data30 + parity_30_from_30(data30)

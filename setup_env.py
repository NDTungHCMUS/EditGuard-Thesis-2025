# This script updates the import in `degradations.py` to fix compatibility
# with newer versions of `torchvision`, where `rgb_to_grayscale` was moved
# from `torchvision.transforms.functional_tensor` to
# `torchvision.transforms.functional`.

import os

# Path to the file
file_path = '/venv/main/lib/python3.12/site-packages/basicsr/data/degradations.py'

# Read file content
with open(file_path, 'r') as file:
    lines = file.readlines()

# Replace the incorrect import
with open(file_path, 'w') as file:
    for line in lines:
        if 'functional_tensor' in line and 'rgb_to_grayscale' in line:
            print(f"🔧 Replacing line: {line.strip()}")
            file.write('from torchvision.transforms.functional import rgb_to_grayscale\n')
        else:
            file.write(line)

print("✅ Patched basicsr/data/degradations.py")
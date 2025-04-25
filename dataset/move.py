import os
from PIL import Image

# Define the source and destination directories.
folder_A = "valAGE-Set-Mask(512,512)"  # Replace with the path to folder A.
folder_B = "valAGE-Set-Mask(4096,3072)"  # Replace with the path to folder B.

# Create folder B if it doesn't exist.
if not os.path.exists(folder_B):
    os.makedirs(folder_B)

# Define allowed image extensions.
allowed_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif')

# Loop through all the files in folder A.
for filename in os.listdir(folder_A):
    if filename.lower().endswith(allowed_extensions):
        src_path = os.path.join(folder_A, filename)
        dst_path = os.path.join(folder_B, filename)

        try:
            # Open the image.
            with Image.open(src_path) as img:
                # Resize the image to 4096 x 3072 using a high-quality downsampling filter.
                resized_img = img.resize((4096, 3072), Image.LANCZOS)
                # Save the resized image to folder B.
                resized_img.save(dst_path)
                print(f"Copied and scaled: {filename}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")

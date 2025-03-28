import random

# ----- VN Start -----
def generate_dataset(num_images, num_strings_per_image, output_filename="../dataset/copyrights-10-eles.txt"):
    lines = []
    for i in range(1, num_images + 1):
        # Format the image number as a 4-digit zero-padded string.
        image_number = f"{i:04d}"
        lines.append(image_number)
            
        # Generate the required number of 64-bit binary strings.
        for _ in range(num_strings_per_image):
            bit_string = ''.join(random.choice("01") for _ in range(64))
            lines.append(bit_string)
    
    # Join the lines without adding a trailing newline at the end.
    file_content = "\n".join(lines)
    
    with open(output_filename, "w") as f:
        f.write(file_content)

if __name__ == "__main__":
    generate_dataset(10, 2)
    
# ----- VN End -----
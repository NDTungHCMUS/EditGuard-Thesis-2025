# ----- VN START -----
import os
import global_variables

RESULT_FILE = global_variables.TEST_CONFIG['result_path']

def ensure_directory_exists(filename):
    """Ensure the directory for the given file exists; create it if not."""
    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

def append_content_to_file(content, filename=RESULT_FILE):
    """Append any arbitrary string content to the specified file."""
    ensure_directory_exists(filename)
    
    with open(filename, "a") as file:
        file.write(content + "\n")

class ImageResult:
    def __init__(self, image_id, ori_message, rec_message, bit_errors, output_file=RESULT_FILE):
        # Enforce that both messages have the same length.
        if len(ori_message) != len(rec_message):
            raise ValueError("Original message and received message must be of the same length.")
        self.image_id = image_id
        self.ori_message = ori_message
        self.rec_message = rec_message
        self.bit_errors = bit_errors
        self.recalculate_errors()
        self.output_file = output_file

    def recalculate_errors(self):
        """Calculate error positions and count by comparing the original and received messages."""
        if len(self.ori_message) != len(self.rec_message):
            raise ValueError("Original message and received message must be of the same length.")
        self.error_positions = [i for i in range(len(self.ori_message))
                                if self.ori_message[i] != self.rec_message[i]]
        self.number_errors_position = len(self.error_positions)

    def to_pretty_string(self):
        """Return a beautifully formatted string representation of the ImageResult.
        
        The header includes the image ID, while the body remains unchanged.
        """
        line_length = 60
        border = "=" * line_length
        label_width = 30  # For alignment
        header = f"Image Result - {self.image_id}"
        pretty_lines = [
            border,
            header.center(line_length),
            border,
            f"{'Image ID'.ljust(label_width)}: {self.image_id}",
            f"{'Message Length'.ljust(label_width)}: {len(self.ori_message)}",
            f"{'Original Message'.ljust(label_width)}: {self.ori_message}",
            f"{'Received Message'.ljust(label_width)}: {self.rec_message}",
            f"{'Bit Errors'.ljust(label_width)}: {self.bit_errors}",
            f"{'Error Count'.ljust(label_width)}: {self.number_errors_position}",
            f"{'Error Positions'.ljust(label_width)}: {self.error_positions}",
            # f"{'Error Positions'.ljust(label_width)}: " +
            #     (", ".join(map(str, self.error_positions)) if self.error_positions else "None"),
            # border
        ]
        return "\n".join(pretty_lines)

    def write_to_file(self, filename=RESULT_FILE):
        """Write the pretty formatted result to the specified file, overwriting any existing content."""
        ensure_directory_exists(filename)

        with open(filename, "w") as file:
            file.write(self.to_pretty_string() + "\n")

    def append_to_file(self, filename=RESULT_FILE):
        """Append the pretty formatted result to the specified file."""
        ensure_directory_exists(filename)

        with open(filename, "a") as file:
            file.write(self.to_pretty_string() + "\n")

# Example usage:
if __name__ == "__main__":
    result = ImageResult(
        image_id="IMG_001",
        ori_message="Hello, Wo3ld!",
        rec_message="Hella, World!",  # Both messages must be of the same length.
        bit_errors=0.05
    )
    
    # Print the pretty formatted result to the console.
    print(result.to_pretty_string())
    
    # Write (overwrite) the result to a file.
    result.write_to_file("result.txt")
    
    # Append the result to the file.
    result.append_to_file("result.txt")

    # Append the result to the file.
    result.append_to_file("result.txt")
# ----- VN END -----
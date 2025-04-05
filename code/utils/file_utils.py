# ----- VN START -----
import os

def clear_file(filepath):
    """Clears the contents of the specified file."""
    with open(filepath, 'w') as file:
        pass  # Opening in 'w' mode automatically clears the file

def ensure_directory_exists(filename):
    """Ensure the directory for the given file exists; create it if not."""
    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

def append_content_to_file(content, filename):
    """Append any arbitrary string content to the specified file."""
    ensure_directory_exists(filename)
    
    with open(filename, "a") as file:
        file.write(content + "\n")

# ----- VN END -----
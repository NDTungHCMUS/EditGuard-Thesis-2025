import random
import yaml
from easydict import EasyDict
import os
from pathlib import Path

def load_opt(cfg_path: str) -> EasyDict:
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return EasyDict(cfg)

def generate_bit_blocks(output_path, num_images=10, bits_per_image=2):
    with open(output_path, 'w') as f:
        for i in range(1, num_images + 1):
            image_id = f"{i:04d}"  # Format as 4-digit string
            f.write(f"{image_id}\n")
            for _ in range(bits_per_image):
                bits = ''.join(random.choice('01') for _ in range(64))
                f.write(f"{bits}\n")
    print(f"Generated file with {num_images} image blocks at {output_path}")

if __name__ == "__main__":
    # 1) Resolve this script’s own directory
    script_dir = Path(__file__).resolve().parent

    # 2) Jump up from dataset/ to project root, then into code/options
    project_root = script_dir.parent
    cfg_path     = project_root / "code" / "options" / "test_editguard.yml"

    # 3) Load config
    opt = load_opt(str(cfg_path))
    print("Loaded config:", opt)

    # 4) Build an output path INSIDE the same folder as this script:
    out_path = script_dir / f"{opt.datasets.TD.copyright_path}"

    # 5) Generate bit-blocks there
    generate_bit_blocks(
        output_path  = str(out_path),
        num_images   = 20,
        bits_per_image = (
            opt.copyright_length//64
          + opt.phash_length//64
          + opt.metadata_length//64
        )
    )
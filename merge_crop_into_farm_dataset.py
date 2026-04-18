from pathlib import Path
import shutil

# Folder that contains  crop and weed dataset
CROP_ROOT = Path("datasets/crop_dataset")

# Folder where our final merged dataset is stored
TARGET_ROOT = Path("datasets/farm_dataset")

# Dataset splits
SPLITS = ["train", "val", "test"]

#  this was used to merge datasets
# Final merged dataset:
# 0 = chicken
# 1 = crop
# 2 = weed
CLASS_MAP = {
    0: 1,
    1: 2,
}

def ensure_dirs():
    # Make sure the output image and label folders exist
    for split in SPLITS:
        (TARGET_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (TARGET_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)

def merge_split(split: str):
    # Source folders dataset
    src_img_dir = CROP_ROOT / "images" / split
    src_lbl_dir = CROP_ROOT / "labels" / split

    # Destination folders in our merged dataset
    dst_img_dir = TARGET_ROOT / "images" / split
    dst_lbl_dir = TARGET_ROOT / "labels" / split

    # Stop if image folder is missing
    if not src_img_dir.exists():
        print(f"Missing image folder: {src_img_dir}")
        return 0

    # Stop if label folder is missing
    if not src_lbl_dir.exists():
        print(f"Missing label folder: {src_lbl_dir}")
        return 0

    # Get all image files in this split
    image_files = [p for p in src_img_dir.iterdir() if p.is_file()]
    copied = 0

    # Go through each image
    for image_path in image_files:
        # Find the matching label file
        src_label_path = src_lbl_dir / f"{image_path.stem}.txt"
        if not src_label_path.exists():
            continue

        # Copy the image into our merged dataset
        shutil.copy2(image_path, dst_img_dir / image_path.name)

        # Read the old labels and convert class numbers
        new_lines = []
        with open(src_label_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()

                # Skip bad lines
                if len(parts) != 5:
                    continue

                old_class, x, y, w, h = parts

                # Try to turn class id into an integer
                try:
                    old_class = int(old_class)
                except ValueError:
                    continue

                # Skip labels we do not know how to map
                if old_class not in CLASS_MAP:
                    continue

                # Change class id
                new_class = CLASS_MAP[old_class]
                new_lines.append(f"{new_class} {x} {y} {w} {h}")

        # Skip if nothing valid was found
        if not new_lines:
            continue

        # Save the new remapped label file
        dst_label_path = dst_lbl_dir / f"{image_path.stem}.txt"
        with open(dst_label_path, "w", encoding="utf-8") as f:
            f.write("\n".join(new_lines))

        copied += 1

    return copied

def main():
    # Make sure output folders exist
    ensure_dirs()

    total = 0

    # Merge train, val, and test one by one
    for split in SPLITS:
        count = merge_split(split)
        total += count
        print(f"{split}: merged {count} crop/weed image(s)")

    # Final summary
    print(f"\nDone. Total merged crop/weed images: {total}")

if __name__ == "__main__":
    main()
from pathlib import Path
import shutil

# Folder that contains our negative images
NEG_ROOT = Path("negative_images")

# Main merged dataset folder
DATASET_ROOT = Path("datasets/farm_dataset")

# Dataset splits
SPLITS = ["train", "val", "test"]


def main():
    # Go through train, val, and test folders
    for split in SPLITS:
        src_dir = NEG_ROOT / split
        dst_img_dir = DATASET_ROOT / "images" / split
        dst_lbl_dir = DATASET_ROOT / "labels" / split

        # Skip this split if the negative image folder does not exist
        if not src_dir.exists():
            print(f"Skipping missing folder: {src_dir}")
            continue

        # Get all files in the current negative image folder
        files = [f for f in src_dir.iterdir() if f.is_file()]
        print(f"\nProcessing {split}: {len(files)} files")

        copied = 0
        for file_path in files:
            # Copy the negative image into the dataset
            dst_img_path = dst_img_dir / file_path.name
            shutil.copy2(file_path, dst_img_path)

            # Create an empty label file because there are no objects to detect
            empty_label_path = dst_lbl_dir / f"{file_path.stem}.txt"
            empty_label_path.write_text("", encoding="utf-8")

            copied += 1

        # Print how many negative images were added
        print(f"Added {copied} negative images to {split}")

    print("\nDone adding negative images.")


if __name__ == "__main__":
    # Start adding negative images
    main()
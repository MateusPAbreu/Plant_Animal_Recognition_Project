from pathlib import Path
import shutil

NEG_ROOT = Path("negative_images")
DATASET_ROOT = Path("datasets/farm_dataset")

SPLITS = ["train", "val", "test"]


def main():
    for split in SPLITS:
        src_dir = NEG_ROOT / split
        dst_img_dir = DATASET_ROOT / "images" / split
        dst_lbl_dir = DATASET_ROOT / "labels" / split

        if not src_dir.exists():
            print(f"Skipping missing folder: {src_dir}")
            continue

        files = [f for f in src_dir.iterdir() if f.is_file()]
        print(f"\nProcessing {split}: {len(files)} files")

        copied = 0
        for file_path in files:
            # Copy image
            dst_img_path = dst_img_dir / file_path.name
            shutil.copy2(file_path, dst_img_path)

            # Create matching empty label file
            empty_label_path = dst_lbl_dir / f"{file_path.stem}.txt"
            empty_label_path.write_text("", encoding="utf-8")

            copied += 1

        print(f"Added {copied} negative images to {split}")

    print("\nDone adding negative images.")


if __name__ == "__main__":
    main()
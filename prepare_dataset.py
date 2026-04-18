from pathlib import Path
import shutil
import random
import cv2

# =========================================================
# CONFIG
# =========================================================

# Original chicken dataset
CHICKEN_SOURCE_IMAGES = Path("datasets/v1_240507/images")
CHICKEN_SOURCE_MASKS = Path("datasets/v1_240507/masks")
CHICKEN_CLASS_ID = 0

# Mateus crop/weed dataset
PLANT_SOURCE_ROOT = Path("datasets/crop_dataset")

# Final merged dataset
TARGET_ROOT = Path("datasets/farm_dataset")

# Output image folders
TRAIN_IMG = TARGET_ROOT / "images" / "train"
VAL_IMG = TARGET_ROOT / "images" / "val"
TEST_IMG = TARGET_ROOT / "images" / "test"

# Output label folders
TRAIN_LBL = TARGET_ROOT / "labels" / "train"
VAL_LBL = TARGET_ROOT / "labels" / "val"
TEST_LBL = TARGET_ROOT / "labels" / "test"

# Split ratios for chicken dataset
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

# Random seed for repeatable results
RANDOM_SEED = 42

# Limit chicken images for faster training
MAX_CHICKEN_IMAGES = 300

# Remap Mateus classes into final merged classes
# Mateus: 0 = crop, 1 = weed
# Final:  1 = crop, 2 = weed
PLANT_CLASS_MAP = {
    0: 1,
    1: 2,
}

SPLITS = ["train", "val", "test"]


# =========================================================
# HELPERS
# =========================================================

def ensure_dirs():
    # Make sure all output folders exist
    for folder in [TRAIN_IMG, VAL_IMG, TEST_IMG, TRAIN_LBL, VAL_LBL, TEST_LBL]:
        folder.mkdir(parents=True, exist_ok=True)


def clear_output_dirs():
    # Remove old files from the merged dataset folders
    for folder in [TRAIN_IMG, VAL_IMG, TEST_IMG, TRAIN_LBL, VAL_LBL, TEST_LBL]:
        if folder.exists():
            for item in folder.iterdir():
                if item.is_file():
                    item.unlink()


def yolo_bbox_from_mask(mask_path, img_width, img_height):
    """
    Read a chicken mask and convert the white object area into
    a YOLO bounding box.
    """
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None

    ys, xs = (mask > 0).nonzero()
    if len(xs) == 0 or len(ys) == 0:
        return None

    x_min = xs.min()
    x_max = xs.max()
    y_min = ys.min()
    y_max = ys.max()

    bbox_width = x_max - x_min + 1
    bbox_height = y_max - y_min + 1
    x_center = x_min + bbox_width / 2
    y_center = y_min + bbox_height / 2

    x_center /= img_width
    y_center /= img_height
    bbox_width /= img_width
    bbox_height /= img_height

    return x_center, y_center, bbox_width, bbox_height


def find_instance_masks_for_image(image_name):
    """
    For image_0.png, find matching mask files like:
    image_0_instanceMask_....png
    """
    stem = Path(image_name).stem
    pattern = f"{stem}_instanceMask*.png"
    return sorted(CHICKEN_SOURCE_MASKS.glob(pattern))


def process_chicken_image(image_path, img_dest, lbl_dest):
    """
    Convert one chicken image and its masks into a YOLO label file.
    """
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Skipping unreadable chicken image: {image_path.name}")
        return False

    h, w = image.shape[:2]
    masks = find_instance_masks_for_image(image_path.name)

    label_lines = []

    for mask_path in masks:
        bbox = yolo_bbox_from_mask(mask_path, w, h)
        if bbox is None:
            continue

        x_center, y_center, bw, bh = bbox
        line = f"{CHICKEN_CLASS_ID} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}"
        label_lines.append(line)

    if not label_lines:
        print(f"No valid masks for chicken image: {image_path.name}")
        return False

    # Copy image
    shutil.copy2(image_path, img_dest / image_path.name)

    # Save label
    label_path = lbl_dest / f"{image_path.stem}.txt"
    with open(label_path, "w", encoding="utf-8") as f:
        f.write("\n".join(label_lines))

    return True


def build_chicken_dataset():
    """
    Build chicken part of the merged dataset from masks.
    """
    image_files = sorted(CHICKEN_SOURCE_IMAGES.glob("*.png"))
    print(f"Found {len(image_files)} chicken images.")

    random.seed(RANDOM_SEED)
    random.shuffle(image_files)
    image_files = image_files[:MAX_CHICKEN_IMAGES]

    total = len(image_files)
    train_count = int(total * TRAIN_RATIO)
    val_count = int(total * VAL_RATIO)
    test_count = total - train_count - val_count

    train_files = image_files[:train_count]
    val_files = image_files[train_count:train_count + val_count]
    test_files = image_files[train_count + val_count:]

    print(f"Chicken split -> Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")

    success_train = 0
    success_val = 0
    success_test = 0

    for img in train_files:
        if process_chicken_image(img, TRAIN_IMG, TRAIN_LBL):
            success_train += 1

    for img in val_files:
        if process_chicken_image(img, VAL_IMG, VAL_LBL):
            success_val += 1

    for img in test_files:
        if process_chicken_image(img, TEST_IMG, TEST_LBL):
            success_test += 1

    print("\nChicken processing done.")
    print(f"Usable chicken train images: {success_train}")
    print(f"Usable chicken val images:   {success_val}")
    print(f"Usable chicken test images:  {success_test}")


def merge_plant_split(split, img_dest, lbl_dest):
    """
    Copy Mateus crop/weed images and remap labels:
    crop 0 -> 1
    weed 1 -> 2
    """
    src_img_dir = PLANT_SOURCE_ROOT / "images" / split
    src_lbl_dir = PLANT_SOURCE_ROOT / "labels" / split

    if not src_img_dir.exists():
        print(f"Missing image folder: {src_img_dir}")
        return 0

    if not src_lbl_dir.exists():
        print(f"Missing label folder: {src_lbl_dir}")
        return 0

    image_files = [p for p in src_img_dir.iterdir() if p.is_file()]
    copied = 0

    for image_path in image_files:
        src_label_path = src_lbl_dir / f"{image_path.stem}.txt"
        if not src_label_path.exists():
            continue

        new_lines = []

        with open(src_label_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue

                old_class, x, y, w, h = parts

                try:
                    old_class = int(old_class)
                except ValueError:
                    continue

                if old_class not in PLANT_CLASS_MAP:
                    continue

                new_class = PLANT_CLASS_MAP[old_class]
                new_lines.append(f"{new_class} {x} {y} {w} {h}")

        if not new_lines:
            continue

        # Copy image
        shutil.copy2(image_path, img_dest / image_path.name)

        # Save remapped label
        dst_label_path = lbl_dest / f"{image_path.stem}.txt"
        with open(dst_label_path, "w", encoding="utf-8") as f:
            f.write("\n".join(new_lines))

        copied += 1

    return copied


def merge_plant_dataset():
    """
    Merge crop and weed dataset into the final dataset.
    """
    plant_train = merge_plant_split("train", TRAIN_IMG, TRAIN_LBL)
    plant_val = merge_plant_split("val", VAL_IMG, VAL_LBL)
    plant_test = merge_plant_split("test", TEST_IMG, TEST_LBL)

    print("\nCrop/weed processing done.")
    print(f"Copied crop/weed train images: {plant_train}")
    print(f"Copied crop/weed val images:   {plant_val}")
    print(f"Copied crop/weed test images:  {plant_test}")


def main():
    # Prepare clean folders
    ensure_dirs()
    clear_output_dirs()

    # Step 1: build chicken dataset
    build_chicken_dataset()

    # Step 2: merge crop/weed dataset
    merge_plant_dataset()

    print("\nFinal merged dataset build complete.")


if __name__ == "__main__":
    main()
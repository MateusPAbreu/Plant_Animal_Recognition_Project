from pathlib import Path
import shutil
import random
import cv2

# -----------------------------
# CONFIG
# -----------------------------
SOURCE_IMAGES = Path("datasets/weed_dataset/images")
# SOURCE_MASKS = Path("datasets/v1_240507/masks")

TARGET_ROOT = Path("datasets/weed_dataset")
TRAIN_IMG = TARGET_ROOT / "images" / "train"
VAL_IMG = TARGET_ROOT / "images" / "val"
TEST_IMG = TARGET_ROOT / "images" / "test"

TRAIN_LBL = TARGET_ROOT / "labels" / "train"
VAL_LBL = TARGET_ROOT / "labels" / "val"
TEST_LBL = TARGET_ROOT / "labels" / "test"

CLASS_ID = 0  # weed

TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

RANDOM_SEED = 42
MAX_IMAGES = 300   # start small for first project version

def ensure_dirs():
    for folder in [TRAIN_IMG, VAL_IMG, TEST_IMG, TRAIN_LBL, VAL_LBL, TEST_LBL]:
        folder.mkdir(parents=True, exist_ok=True)


def yolo_bbox_from_mask(img_width, img_height):
    """
    Read a binary instance mask and convert nonzero pixels to YOLO bbox.
    Returns: (x_center, y_center, width, height) normalized, or None if empty.
    """
    mask = cv2.imread( cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None

    # Find non-black pixels
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

    # Normalize
    x_center /= img_width
    y_center /= img_height
    bbox_width /= img_width
    bbox_height /= img_height

    return x_center, y_center, bbox_width, bbox_height


def find_instance_masks_for_image(image_name):
    """
    For image_0.png, find masks like:
    image_0_instanceMask_....png
    Ignore segmentationMask files.
    """
    stem = Path(image_name).stem
    pattern = f"{stem}_instanceMask*.png"
    return sorted(SOURCE_IMAGES.glob(pattern))


def process_image(image_path, img_dest, lbl_dest):
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Skipping unreadable image: {image_path.name}")
        return False

    h, w = image.shape[:2]
    masks = find_instance_masks_for_image(image_path.name)

    label_lines = []
    for mask_path in masks:
        bbox = yolo_bbox_from_mask(w, h)
        if bbox is None:
            continue

        x_center, y_center, bw, bh = bbox
        line = f"{CLASS_ID} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}"
        label_lines.append(line)

    # Skip image if it has no usable masks
    if not label_lines:
        print(f"No valid masks for: {image_path.name}")
        return False

    shutil.copy2(image_path, img_dest / image_path.name)

    label_path = lbl_dest / f"{image_path.stem}.txt"
    with open(label_path, "w", encoding="utf-8") as f:
        f.write("\n".join(label_lines))

    return True


def main():
    ensure_dirs()
    #likely have to "fix" source images
    image_files = sorted(SOURCE_IMAGES.glob("*.jpg"))
    print(f"Found {len(image_files)} images.")

    # start with a smaller subset
    random.seed(RANDOM_SEED)
    random.shuffle(image_files)
    image_files = image_files[:MAX_IMAGES]

    total = len(image_files)
    train_count = int(total * TRAIN_RATIO)
    val_count = int(total * VAL_RATIO)
    test_count = total - train_count - val_count

    train_files = image_files[:train_count]
    val_files = image_files[train_count:train_count + val_count]
    test_files = image_files[train_count + val_count:]

    print(f"Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")

    success_train = 0
    success_val = 0
    success_test = 0

    for img in train_files:
        if process_image(img, TRAIN_IMG):
            success_train += 1

    for img in val_files:
        if process_image(img, VAL_IMG):
            success_val += 1

    for img in test_files:
        if process_image(img, TEST_IMG):
            success_test += 1

    print("\nDone.")
    print(f"Usable train images: {success_train}")
    print(f"Usable val images:   {success_val}")
    print(f"Usable test images:  {success_test}")


if __name__ == "__main__":
    main()
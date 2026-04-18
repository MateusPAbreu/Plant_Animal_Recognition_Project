from ultralytics import YOLO
from pathlib import Path

# Path to the trained model we want to test
MODEL_PATH = r"C:\Users\david\runs\detect\agriguard_chicken_crop_weed_model\weights\best.pt"

# Folder that contains test images
IMAGE_FOLDER = "test_images"

# Class id to class name mapping
CLASS_NAMES = {
    0: "chicken",
    1: "crop",
    2: "weed"
}

def main():
    # Load the trained model
    model = YOLO(MODEL_PATH)

    # Run prediction on all images in the test_images folder
    results = model.predict(
        source=IMAGE_FOLDER,
        save=True,       # save images with predicted boxes
        conf=0.50,       # confidence threshold
        verbose=False    # keep terminal output cleaner
    )

    print("\nPrediction results:")
    print("-" * 60)

    # Go through each prediction result
    for result in results:
        image_name = Path(result.path).name

        # Keep count of each detected class
        counts = {
            "chicken": 0,
            "crop": 0,
            "weed": 0
        }

        # Check if the model found any boxes
        if result.boxes is not None and len(result.boxes) > 0:
            classes = result.boxes.cls.tolist()

            # Count how many times each class appears
            for cls_id in classes:
                cls_id = int(cls_id)
                if cls_id in CLASS_NAMES:
                    counts[CLASS_NAMES[cls_id]] += 1

        # Print the result for this image
        print(
            f"{image_name}: "
            f"{counts['chicken']} chicken(s), "
            f"{counts['crop']} crop(s), "
            f"{counts['weed']} weed(s)"
        )

    print("-" * 60)
    print("Prediction complete.")
    print("Saved images are in the latest runs/detect/predict folder.")

if __name__ == "__main__":
    # Start the prediction process
    main()
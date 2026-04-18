from ultralytics import YOLO
from pathlib import Path

MODEL_PATH = r"C:\Users\ashla\Documents\UNBC\CPSC 371\Prodj\Plant_Animal_Recognition_Project\runs\detect\agriguard_cow_model_v2\weights\best.pt"
IMAGE_FOLDER = "test_images"

def main():
    model = YOLO(MODEL_PATH)

    results = model.predict(
        source=IMAGE_FOLDER,
        save=True,
        conf=0.60,
        verbose=False
    )

    print("\nPrediction results:")
    print("-" * 50)

    for result in results:
        image_name = Path(result.path).name
        cow_count = len(result.boxes) if result.boxes is not None else 0
        print(f"{image_name}: {cow_count} Cows(s) detected")

    print("-" * 50)
    print("Prediction complete.")
    print("Saved images are in the latest runs/detect/predict folder.")

if __name__ == "__main__":
    main()
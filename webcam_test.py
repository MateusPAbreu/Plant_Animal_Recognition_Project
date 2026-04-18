from ultralytics import YOLO
import cv2

# Path to the final trained model for chicken, crop, and weed
MODEL_PATH = r"C:\Users\david\runs\detect\agriguard_chicken_crop_weed_model\weights\best.pt"

# Camera index for the webcam
CAMERA_INDEX = 0

# Map class numbers to readable names
CLASS_NAMES = {
    0: "chicken",
    1: "crop",
    2: "weed"
}

def main():
    # Load the trained model
    model = YOLO(MODEL_PATH)

    # Open the webcam
    cap = cv2.VideoCapture(CAMERA_INDEX)

    # Stop if the camera cannot be opened
    if not cap.isOpened():
        print(f"Could not open camera index {CAMERA_INDEX}")
        return

    print("Camera started. Press Q to quit.")

    while True:
        # Read one frame from the webcam
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from camera.")
            break

        # Run prediction on the current frame
        results = model.predict(frame, conf=0.50, verbose=False)
        result = results[0]

        # Start counters for each class
        counts = {
            "chicken": 0,
            "crop": 0,
            "weed": 0
        }

        # Count detected objects by class
        if result.boxes is not None and len(result.boxes) > 0:
            classes = result.boxes.cls.tolist()
            for cls_id in classes:
                cls_id = int(cls_id)
                if cls_id in CLASS_NAMES:
                    counts[CLASS_NAMES[cls_id]] += 1

        # Draw detection boxes on the frame
        annotated_frame = result.plot()

        # Show counts on the screen
        cv2.putText(
            annotated_frame,
            f"Chickens: {counts['chicken']}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            annotated_frame,
            f"Crops: {counts['crop']}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            annotated_frame,
            f"Weeds: {counts['weed']}",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        # Show the live webcam window
        cv2.imshow("AgriGuard Live Detection", annotated_frame)

        # Press Q to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release the camera and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Start live webcam detection
    main()
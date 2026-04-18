from ultralytics import YOLO
import cv2

MODEL_PATH = r"C:\Users\ashla\Documents\UNBC\CPSC 371\Prodj\Plant_Animal_Recognition_Project\runs\detect\agriguard_cow_model_v1\weights\best.pt"
CAMERA_INDEX = 0

def main():
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print(f"Could not open camera index {CAMERA_INDEX}")
        return

    print("Camera started. Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from camera.")
            break

        results = model.predict(frame, conf=0.60, verbose=False)
        result = results[0]

        chicken_count = len(result.boxes) if result.boxes is not None else 0
        annotated_frame = result.plot()

        cv2.putText(
            annotated_frame,
            f"Chickens detected: {chicken_count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.imshow("AgriGuard Live Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
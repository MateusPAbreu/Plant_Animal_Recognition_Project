from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")

    model.train(
        data="datasets/cow_dataset/data.yaml",
        epochs=300,
        imgsz=640,
        batch=8,
        name="agriguard_cow_model_v2"
    )

if __name__ == "__main__":
    main()
from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")

    model.train(
        data="datasets/farm_dataset/data.yaml",
        epochs=30,
        imgsz=640,
        batch=8,
        name="agriguard_chicken_model_v2"
    )

if __name__ == "__main__":
    main()
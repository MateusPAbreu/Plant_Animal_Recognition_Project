from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt") #it was 8 before

    model.train(
        data="datasets/weed_dataset/data.yaml",
        epochs=30, #30 before
        imgsz=640,
        batch=8,
        name="agriguard_weed_model_v2"
    )

if __name__ == "__main__":
    main()
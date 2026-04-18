from ultralytics import YOLO

def main():
    # Load the base YOLOv8 nano model
    model = YOLO("yolov8n.pt")

    # Train the model using our merged farm dataset
    model.train(
        data="datasets/farm_dataset/data.yaml",   # dataset config file
        epochs=30,                                # number of training rounds
        imgsz=640,                                # image size used for training
        batch=8,                                  # number of images per batch
        name="agriguard_chicken_crop_weed_model" # name of the training run
    )

if __name__ == "__main__":
    # Start training
    main()
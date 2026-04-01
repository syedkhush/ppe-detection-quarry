from ultralytics import YOLO

def train():
    model = YOLO('yolov8n.pt')
    model.train(
        data='data.yaml',
        epochs=30,
        imgsz=640,
        project='runs/detect',
        name='ppe_detection'
    )
    print("Training complete!")

if __name__ == "__main__":
    train()

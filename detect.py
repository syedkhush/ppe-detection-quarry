from ultralytics import YOLO
import sys

def detect(image_path, model_path='runs/detect/ppe_detection/weights/best.pt'):
    model = YOLO(model_path)
    results = model.predict(
        source=image_path,
        save=True,
        conf=0.5,
        project='results',
        name='detection'
    )
    for r in results:
        print(f"Detected: {r.boxes.cls.tolist()}")
        print(f"Confidence: {r.boxes.conf.tolist()}")

if __name__ == "__main__":
    image = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"
    detect(image)

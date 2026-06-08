from ultralytics import YOLO

def train():
    model = YOLO("yolo26n.pt")

    model.train(
        data="datasets/vehicles/data.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        workers=2,
        device=0,
    )

if __name__ == "__main__":
    train()
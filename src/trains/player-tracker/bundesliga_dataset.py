from ultralytics import YOLO


def train():
    # Load model
    model = YOLO("yolo26n.pt")

    # Train model
    model.train(
        data="datasets/football-player-tracker/data.yaml",
        epochs=150,
        imgsz=960,  # Recommended larger resolution for small/far objects or high resolution images/videos (1024, 1280, ...)
        batch=16,
        workers=8,
        device=0,
    )

    # For best results on ball detection, consider test-time augmentation (TTA)
    # or evaluating at imgsz=2560.


if __name__ == "__main__":
    train()

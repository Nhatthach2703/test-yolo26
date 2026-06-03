from ultralytics import YOLO

def detect_video():
    # load model
    model = YOLO("runs/detect/train/weights/best.pt")

    # run inference
    model.predict(
        source="images/SampleVideo_LowQuality.mp4",   # video input
        show=True,           # show realtime window
        save=True,           # save output video
        conf=0.25,           # confidence threshold
        device=0,            # GPU
    )

if __name__ == "__main__":
    detect_video()
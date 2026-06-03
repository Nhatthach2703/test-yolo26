import cv2
from ultralytics import YOLO


def detect_video():
    model = YOLO("runs/detect/train/weights/best.pt")

    results = model.predict(
        source="images/test-2.mp4", stream=True, conf=0.25, device=0  # quan trọng
    )

    cv2.namedWindow("YOLO", cv2.WINDOW_NORMAL)  # resize được
    cv2.resizeWindow("YOLO", 1280, 720)

    for r in results:
        frame = r.plot()

        cv2.imshow("YOLO", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    detect_video()

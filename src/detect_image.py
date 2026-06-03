from ultralytics import YOLO

# Load pretrained model
model = YOLO("models/yolo26n.pt")

# Predict
results = model("images/dog-2.jpg")

for result in results:
    boxes = result.boxes

    for box in boxes:
        cls = int(box.cls[0])

        print(
            model.names[cls]
        )
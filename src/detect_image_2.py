from ultralytics import YOLO
import cv2

# Load pretrained model
model = YOLO("models/yolo26n.pt")

# Predict
results = model("images/snow-leopard.jpg")

for result in results:
    # Get annotated image with bounding boxes
    annotated_img = result.plot()
    
    # Display image using OpenCV
    cv2.imshow("YOLO Detection Result", annotated_img)
    cv2.waitKey(0) # Wait for any key press to close
    cv2.destroyAllWindows()
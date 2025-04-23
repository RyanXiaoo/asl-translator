import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("models/best_my_best.h5")
IMG_SIZE = (128, 128)

# ASL alphabet labels
labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ['del', 'nothing', 'space']

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Draw a box to show region of interest (ROI)
    x1, y1, x2, y2 = 100, 100, 400, 400
    roi = frame[y1:y2, x1:x2]

    # Preprocess ROI
    img = cv2.resize(roi, IMG_SIZE)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    preds = model.predict(img)
    class_idx = np.argmax(preds[0])
    confidence = np.max(preds[0])

    label = f"{labels[class_idx]} ({confidence:.2f})"

    # Display box and label
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    cv2.imshow("ASL Real-Time Prediction", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

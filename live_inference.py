import cv2, numpy as np
from collections import deque
from model import build_model

IMG_SIZE = (128,128)
WEIGHTS  = "models/best_my_best.h5"
DATA_DIR = "data/asl_alphabet_train/asl_alphabet_train"

# sanity‐check your class names
import tensorflow as tf
ds = tf.keras.utils.image_dataset_from_directory(DATA_DIR, labels="inferred",
    label_mode="categorical", image_size=IMG_SIZE, batch_size=32)
print("classes:", ds.class_names)   # should list A, B, C, … Z, del, nothing, space

# rebuild + load
model = build_model(input_shape=IMG_SIZE+(3,), num_classes=len(ds.class_names))
model.load_weights(WEIGHTS)

buf = deque(maxlen=5)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

while True:
    ret, frame = cap.read()
    if not ret: break

    x1,y1,x2,y2 = 100,100,400,400
    # crop + resize + BGR→RGB + normalize
    roi = cv2.resize(frame[y1:y2, x1:x2], IMG_SIZE)
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)    # critical!
    img = (roi.astype("float32")/255.0)[None,...]

    preds = model.predict(img)[0]
    buf.append(preds)
    avg = np.mean(buf, axis=0)

    idx, conf = avg.argmax(), avg.max()
    label = "…" if conf < 0.7 else ds.class_names[idx]
    if label in ("J","Z"):
        label += " (motion!)"

    cv2.rectangle(frame, (x1,y1),(x2,y2),(255,255,255),2)
    cv2.putText(frame, f"{label} {conf:.2f}", (x1,y1-10),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2)

    cv2.imshow("ASL Live", frame)
    if cv2.waitKey(1)&0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

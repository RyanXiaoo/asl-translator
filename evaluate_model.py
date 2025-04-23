import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.data import AUTOTUNE
from model import build_model

IMG_SIZE = (128,128)
WEIGHTS  = "models/best_my_best.h5"
DATA_DIR = "data/asl_alphabet_train/asl_alphabet_train"

# 1) load raw validation split
raw_val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIR,
    labels="inferred",
    label_mode="categorical",
    image_size=IMG_SIZE,
    batch_size=32,
    validation_split=0.2,
    subset="validation",
    seed=123,
)
class_names = raw_val_ds.class_names
num_classes = len(class_names)

# 2) normalize
val_ds = raw_val_ds.map(
    lambda x, y: (tf.cast(x, tf.float32) / 255.0, y),
    num_parallel_calls=AUTOTUNE,
)

# 3) rebuild + load
model = build_model(input_shape=IMG_SIZE + (3,), num_classes=num_classes)
model.load_weights(WEIGHTS)

# 4) compile & evaluate
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
loss, acc = model.evaluate(val_ds, verbose=1)
print(f"Validation loss: {loss:.4f},  accuracy: {acc:.4f}")

# 5) visualize a few predictions
for images, labels in val_ds.take(1):
    preds = np.argmax(model.predict(images), axis=1)
    truth = np.argmax(labels.numpy(), axis=1)
    plt.figure(figsize=(12,4))
    for i in range(8):
        ax = plt.subplot(2,4,i+1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(f"P: {class_names[preds[i]]}\nT: {class_names[truth[i]]}")
        plt.axis("off")
    plt.show()
    break

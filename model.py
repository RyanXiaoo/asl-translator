import os, argparse
import logging

# 0 = all logs, 1 = filter out INFO, 2 = filter out INFO & WARNING, 3 = filter out all but ERROR
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tensorflow.data import AUTOTUNE
from tensorflow.keras.optimizers.schedules import CosineDecay
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2

# ───────────────────────────────────────────────────────
# Parse CLI arguments
# ───────────────────────────────────────────────────────
HERE            = os.path.dirname(__file__)
PROJECT_ROOT    = os.path.abspath(os.path.join(HERE, "..", ".."))
DEFAULT_DATA_DIR = os.path.join(
    PROJECT_ROOT, "data", "asl_alphabet_train", "asl_alphabet_train"
)



def enable_gpu_growth():
    # Prevent TensorFlow from allocating all GPU memory upfront
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


def get_datasets(data_dir, img_size=(128,128), batch_size=32):
    # 1) load the raw DirectoryDatasets
    raw_train = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        labels='inferred',
        label_mode='categorical',
        image_size=img_size,
        batch_size=batch_size,
        validation_split=0.2,
        subset='training',
        seed=123,
    )
    raw_val = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        labels='inferred',
        label_mode='categorical',
        image_size=img_size,
        batch_size=batch_size,
        validation_split=0.2,
        subset='validation',
        seed=123,
    )

    # 2) grab class names before you strip them off
    class_names = raw_train.class_names

    # 3) data‐augmentation pipeline
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.Resizing(*img_size),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.2),
        tf.keras.layers.RandomTranslation(0.2, 0.2),
        tf.keras.layers.RandomContrast(0.2),
        tf.keras.layers.RandomBrightness(0.2),
        tf.keras.layers.Rescaling(1./255),
    ])

    # 4) apply augmentation to the training set
    train_ds = (
        raw_train
        .map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)
        .shuffle(6000)
        .prefetch(AUTOTUNE)
    )

    # 5) normalize the validation set
    val_ds = (
        raw_val
        .map(lambda x, y: (tf.cast(x, tf.float32)/255.0, y), num_parallel_calls=AUTOTUNE)
        .cache()
        .prefetch(AUTOTUNE)
    )

    return train_ds, val_ds, class_names


def build_model(input_shape=(128,128,3), num_classes=29):
    # 1) load the ImageNet‐pretrained base, drop its top
    base = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )
    base.trainable = False   # freeze for initial training

    # 2) build your new classification head
    x = layers.Input(shape=input_shape)
    y = base(x, training=False)
    y = layers.Dropout(0.3)(y)
    y = layers.Dense(256, activation='relu')(y)
    y = layers.BatchNormalization()(y)
    y = layers.Dropout(0.5)(y)
    outputs = layers.Dense(num_classes, activation='softmax', dtype='float32')(y)

    return Model(inputs=x, outputs=outputs)



def plot_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history['accuracy'], label='train')
    axes[0].plot(history.history['val_accuracy'], label='val')
    axes[0].set_title('Accuracy')
    axes[0].legend()

    axes[1].plot(history.history['loss'], label='train')
    axes[1].plot(history.history['val_loss'], label='val')
    axes[1].set_title('Loss')
    axes[1].legend()

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Train ASL alphabet recognition model")
    parser.add_argument('--data-dir', type=str, default='../data/asl_alphabet_train/asl_alphabet_train', help='Path to training images')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--model-out', type=str, default='models/sign_model.h5')
    args = parser.parse_args()

    enable_gpu_growth()

    train_ds, val_ds, class_names = get_datasets(data_dir=args.data_dir, batch_size=args.batch_size)
    num_classes = len(class_names)

    n_train = tf.data.experimental.cardinality(train_ds).numpy() * args.batch_size
    n_val   = tf.data.experimental.cardinality(val_ds).numpy() * args.batch_size
    print(f"Training on ~{n_train} images, validating on ~{n_val} images")

    steps_per_epoch = n_train // args.batch_size  
    total_steps     = steps_per_epoch * args.epochs

    schedule = CosineDecay(
        initial_learning_rate=1e-3, 
        decay_steps=total_steps,        # ≈ total training steps
        alpha=1e-4               # final LR ≈ 1e-7
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=schedule)

    model = build_model(input_shape=(128,128,3), num_classes=num_classes)

    for layer in model.layers[1].layers[-20:]:
        layer.trainable = True

    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
        ModelCheckpoint('best_' + os.path.basename(args.model_out), monitor='val_loss', save_best_only=True, verbose=1),
        # ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    model.save(args.model_out)

        # after model.save(args.model_out)

    out_dir = os.path.dirname(args.model_out)          # e.g. "notebook/models"
    os.makedirs(out_dir, exist_ok=True)

    # 1) Accuracy plot
    acc_path = os.path.join(out_dir, "accuracy.png")
    plt.figure(figsize=(8, 4))
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    print(f"Saving accuracy plot → {acc_path}")
    plt.savefig(acc_path)
    plt.show()

    # 2) Loss plot
    loss_path = os.path.join(out_dir, "loss.png")
    plt.figure(figsize=(8, 4))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    print(f"Saving loss plot     → {loss_path}")
    plt.savefig(loss_path)
    plt.show()




if __name__ == "__main__":
    main()

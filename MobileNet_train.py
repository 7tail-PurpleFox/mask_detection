import argparse
import os
from pathlib import Path

import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import EarlyStopping

layers = keras.layers

def build_dataset(data_dir, img_size=(224, 224), val_split=0.2, batch_size=32, seed=9527):
    """Create streaming tf.data datasets from directory.

    Returns: train_ds, val_ds, class_names
    """
    data_dir = str(data_dir)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=val_split,
        subset="training",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="int",
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=val_split,
        subset="validation",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="int",
    )

    # get class names BEFORE mapping/caching (image_dataset_from_directory provides it)
    class_names = train_ds.class_names

    # apply MobileNetV2 preprocessing (same semantics as previous code)
    preprocess = keras.applications.mobilenet_v2.preprocess_input
    train_ds = train_ds.map(lambda x, y: (preprocess(tf.cast(x, tf.float32)), y), num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(lambda x, y: (preprocess(tf.cast(x, tf.float32)), y), num_parallel_calls=tf.data.AUTOTUNE)

    train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_ds, val_ds, class_names


def build_model(num_classes, img_size=(224, 224)):
    # Base model: MobileNetV2
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(img_size[0], img_size[1], 3),
        include_top=False,
        weights="imagenet",
    )

    headModel = base_model.output
    headModel = layers.AveragePooling2D(pool_size=(7, 7))(headModel)
    headModel = layers.Flatten(name="flatten")(headModel)
    headModel = layers.Dense(128, activation="relu")(headModel)
    headModel = layers.Dropout(0.5)(headModel)
    headModel = layers.Dense(num_classes, activation="softmax")(headModel)

    model = keras.Model(inputs=base_model.input, outputs=headModel)   
    
    for layer in base_model.layers:
        layer.trainable = False

    return model


def parse_args():
    parser = argparse.ArgumentParser(description="Train MobileNetV2-based mask detector")
    parser.add_argument("--data_dir", type=str, default="./mask_data/clips", help="dataset root directory")
    parser.add_argument("--img_size", type=int, nargs=2, default=(224, 224), help="image size (h w)")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--epochs", type=int, default=20, help="number of epochs to train")
    parser.add_argument("--model_out", type=str, default="mask_detector_mobilenet.h5", help="output model filename")
    parser.add_argument("--plots_dir", type=str, default="./training_plots", help="directory to save training plots and reports")
    return parser.parse_args()


def main():
    args = parse_args()
    # enable GPU memory growth to avoid TF allocating all memory at once
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for g in gpus:
                tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise SystemExit(f"Data directory does not exist: {data_dir}")

    # build streaming datasets
    train_ds, val_ds, class_names = build_dataset(
        data_dir, img_size=tuple(args.img_size), val_split=0.2, batch_size=args.batch_size
    )

    num_classes = len(class_names)
    print(f"Detected classes: {class_names} (#{num_classes})")

    model = build_model(num_classes, img_size=tuple(args.img_size))

    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    lrr = ReduceLROnPlateau(monitor="val_loss", patience=8, verbose=1, factor=0.5, min_lr=1e-6)
    early_stopping = EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)

    # compute class counts from directory to derive class_weight (robust)
    class_counts = []
    for cname in class_names:
        p = data_dir / cname
        cnt = 0
        if p.exists():
            for root, _, files in os.walk(p):
                for f in files:
                    if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                        cnt += 1
        class_counts.append(cnt)

    total = float(sum(class_counts)) if sum(class_counts) > 0 else 1.0
    eps = 1e-6
    class_weight = {}
    max_weight = 10.0
    for i, c in enumerate(class_counts):
        w = total / (len(class_counts) * (c + eps))
        w = min(w, max_weight)
        class_weight[i] = float(w)

    print(f"Class counts: {dict(zip(class_names, class_counts))}")
    print(f"Using class_weight (clipped): {class_weight}")

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=[lrr, early_stopping],
        class_weight=class_weight,
    )
    
    model.save(args.model_out)
    print(f"Saved model to {args.model_out}")

    # Save training plots and evaluation reports
    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # save history as json
    hist_path = plots_dir / "history.json"
    serializable_history = {}
    for k, v in history.history.items():
        try:
            serializable_history[k] = [float(x) for x in v]
        except Exception:
            # fallback: convert items to strings
            serializable_history[k] = [str(x) for x in v]
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(serializable_history, f, ensure_ascii=False, indent=2)

    # plot loss
    plt.figure(figsize=(8, 5))
    plt.plot(history.history.get("loss", []), label="train_loss")
    plt.plot(history.history.get("val_loss", []), label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.title("Loss")
    plt.grid(True)
    plt.savefig(plots_dir / "loss.png")
    plt.close()

    # plot accuracy if available
    if "accuracy" in history.history:
        plt.figure(figsize=(8, 5))
        plt.plot(history.history.get("accuracy", []), label="train_acc")
        plt.plot(history.history.get("val_accuracy", []), label="val_acc")
        plt.xlabel("epoch")
        plt.ylabel("accuracy")
        plt.legend()
        plt.title("Accuracy")
        plt.grid(True)
        plt.savefig(plots_dir / "accuracy.png")
        plt.close()

    # Evaluate on validation set and save confusion matrix + report

    # gather true labels from val_ds
    y_true_list = []
    for _, y in val_ds:
        y_true_list.append(y.numpy())
    if len(y_true_list):
        y_true = np.concatenate(y_true_list, axis=0)
    else:
        y_true = np.array([])

    # predictions on validation dataset
    y_pred_probs = model.predict(val_ds)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # confusion matrix
    try:
        cm = confusion_matrix(y_true, y_pred)
    except Exception:
        cm = None

    if cm is not None:
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        fmt = 'd' if np.issubdtype(cm.dtype, np.integer) else '.2f'
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.tight_layout()
        plt.savefig(plots_dir / "confusion_matrix.png")
        plt.close()

        # save matrix as numpy
        np.save(plots_dir / "confusion_matrix.npy", cm)

        # classification report
        try:
            report = classification_report(y_true, y_pred, target_names=class_names)
        except Exception:
            report = "classification report generation failed"

        with open(plots_dir / "classification_report.txt", "w", encoding="utf-8") as f:
            f.write(report)

if __name__ == "__main__":
    main()

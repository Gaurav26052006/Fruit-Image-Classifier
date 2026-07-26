from pathlib import Path
from random import randint, uniform

import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Rescaling


BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "fruit_cnn.keras"

IMAGE_SIZE = (100, 100)
BATCH_SIZE = 16
EPOCHS = 8
SEED = 42
CLASS_NAMES = ["Apple", "Banana", "Orange"]


def create_demo_dataset(samples_per_class=40):
    """
    Create a tiny synthetic fruit dataset.

    Replace these generated images with real fruit photos for a better model:
    dataset/Apple/*.jpg, dataset/Banana/*.jpg, dataset/Orange/*.jpg
    """
    DATASET_DIR.mkdir(exist_ok=True)

    if any((DATASET_DIR / class_name).glob("*") for class_name in CLASS_NAMES if (DATASET_DIR / class_name).exists()):
        print("Dataset already contains images. Skipping demo dataset generation.")
        return

    print("No dataset found. Creating a small demo dataset...")
    for class_name in CLASS_NAMES:
        class_dir = DATASET_DIR / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        for index in range(samples_per_class):
            image = Image.new("RGB", IMAGE_SIZE, color=(245, 247, 250))
            draw = ImageDraw.Draw(image)

            if class_name == "Apple":
                color = (randint(180, 235), randint(25, 65), randint(25, 65))
                x, y = randint(24, 34), randint(23, 33)
                draw.ellipse((x, y, x + 45, y + 45), fill=color)
                draw.rectangle((x + 22, y - 7, x + 27, y + 7), fill=(90, 55, 25))
                draw.ellipse((x + 28, y - 10, x + 43, y + 2), fill=(45, 140, 55))
            elif class_name == "Banana":
                color = (randint(225, 255), randint(195, 235), randint(35, 80))
                points = [(20, 62), (38, 28), (78, 31), (62, 49), (36, 70)]
                shifted = [(px + randint(-3, 3), py + randint(-3, 3)) for px, py in points]
                draw.pieslice((18, 12, 88, 88), start=20, end=155, fill=color)
                draw.polygon(shifted, fill=(245, 247, 250))
            else:
                color = (randint(230, 255), randint(105, 155), randint(15, 45))
                x, y = randint(23, 33), randint(23, 33)
                draw.ellipse((x, y, x + 48, y + 48), fill=color)
                for _ in range(10):
                    px = randint(x + 8, x + 40)
                    py = randint(y + 8, y + 40)
                    draw.point((px, py), fill=(200, 85, 25))

            angle = uniform(-12, 12)
            image = image.rotate(angle, fillcolor=(245, 247, 250))
            image.save(class_dir / f"{class_name.lower()}_{index + 1:03d}.jpg", quality=92)


def load_datasets():
    """Load images, resize them, normalize pixels, and split into train/test sets."""
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=0.2,
        subset="training",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=0.2,
        subset="validation",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
    )

    class_names = train_ds.class_names
    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(100).prefetch(buffer_size=autotune)
    test_ds = test_ds.cache().prefetch(buffer_size=autotune)
    return train_ds, test_ds, class_names


def build_model(num_classes):
    """Build a simple CNN with Conv2D, MaxPooling2D, Flatten, Dense, and Softmax."""
    return Sequential(
        [
            tf.keras.Input(shape=(*IMAGE_SIZE, 3)),
            Rescaling(1.0 / 255),
            Conv2D(32, (3, 3), activation="relu"),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation="relu"),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation="relu"),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(128, activation="relu"),
            Dense(num_classes, activation="softmax"),
        ]
    )


def main():
    tf.keras.utils.set_random_seed(SEED)
    create_demo_dataset()
    train_ds, test_ds, class_names = load_datasets()

    model = build_model(num_classes=len(class_names))
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()
    model.fit(train_ds, validation_data=test_ds, epochs=EPOCHS)

    loss, accuracy = model.evaluate(test_ds)
    print(f"Test accuracy: {accuracy:.2%}")

    MODEL_DIR.mkdir(exist_ok=True)
    model.save(MODEL_PATH)
    np.save(MODEL_DIR / "class_names.npy", np.array(class_names))
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()

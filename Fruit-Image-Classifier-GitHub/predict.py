from pathlib import Path

import numpy as np
import tensorflow as tf


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "fruit_cnn.keras"
CLASS_NAMES_PATH = BASE_DIR / "model" / "class_names.npy"
IMAGE_SIZE = (100, 100)


def load_model_and_classes():
    """Load the trained Keras model and class labels from disk."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model not found. Run `python train.py` first.")
    if not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError("Class names not found. Run `python train.py` first.")

    model = tf.keras.models.load_model(MODEL_PATH)
    class_names = np.load(CLASS_NAMES_PATH, allow_pickle=True).tolist()
    return model, class_names


def preprocess_image(image_path):
    """Resize the image and convert it into a model-ready batch."""
    image = tf.keras.utils.load_img(image_path, target_size=IMAGE_SIZE)
    image_array = tf.keras.utils.img_to_array(image)
    image_batch = tf.expand_dims(image_array, axis=0)
    return image_batch


def predict_fruit(image_path):
    """Predict the fruit class and return the predicted label and confidence."""
    model, class_names = load_model_and_classes()
    image_batch = preprocess_image(image_path)
    predictions = model.predict(image_batch, verbose=0)[0]
    predicted_index = int(np.argmax(predictions))
    confidence = float(predictions[predicted_index])
    return class_names[predicted_index], round(confidence * 100, 2)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Predict a fruit image class.")
    parser.add_argument("image_path", help="Path to the fruit image")
    args = parser.parse_args()

    label, score = predict_fruit(args.image_path)
    print(f"Prediction: {label}")
    print(f"Confidence: {score}%")

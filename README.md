# Fruit Image Classifier

A simple Python fruit image classifier built with TensorFlow/Keras CNN layers and a Flask upload interface.

The project supports three classes:

- Apple
- Banana
- Orange

If `dataset/` is empty, `train.py` creates a tiny synthetic demo dataset so the project can run immediately. For better accuracy, replace it with real fruit images organized by class.

## Project Structure

```text
Fruit-Image-Classifier/
|-- app.py
|-- train.py
|-- predict.py
|-- dataset/
|   |-- Apple/
|   |-- Banana/
|   |-- Orange/
|-- model/
|-- templates/
|   |-- index.html
|-- static/
|   |-- style.css
|   |-- uploads/
|-- requirements.txt
|-- README.md
```

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```bash
.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset Format

Use this folder layout for real images:

```text
dataset/
|-- Apple/
|   |-- apple1.jpg
|-- Banana/
|   |-- banana1.jpg
|-- Orange/
|   |-- orange1.jpg
```

Images are resized to `100x100` during preprocessing. Pixel values are normalized inside the CNN using a Keras `Rescaling` layer.

## Train the Model

```bash
python train.py
```

This will:

- create a small demo dataset if needed
- split the dataset into training and testing sets
- train a CNN model
- evaluate test accuracy
- save the model to `model/fruit_cnn.keras`
- save class labels to `model/class_names.npy`

## Predict from the Command Line

```bash
python predict.py path/to/fruit-image.jpg
```

## Run the Flask App

```bash
python app.py
```

Open your browser at:

```text
http://127.0.0.1:5000
```

Upload an image to see the predicted fruit class and confidence score.

## Notes

The generated demo dataset is intentionally small and simple. It is useful for checking that the code works, but a real photo dataset will produce a more reliable classifier.

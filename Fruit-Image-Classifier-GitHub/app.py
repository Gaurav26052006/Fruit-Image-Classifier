import os
from pathlib import Path

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

from predict import MODEL_PATH, predict_fruit


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_DIR
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def allowed_file(filename):
    """Return True when the uploaded file has an image extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    image_path = None
    error = None

    if request.method == "POST":
        file = request.files.get("image")

        if not file or file.filename == "":
            error = "Please choose an image to upload."
        elif not allowed_file(file.filename):
            error = "Please upload a PNG, JPG, JPEG, or WEBP image."
        elif not MODEL_PATH.exists():
            error = "Model not found. Run `python train.py` first."
        else:
            filename = secure_filename(file.filename)
            saved_path = UPLOAD_DIR / filename
            file.save(saved_path)

            prediction, confidence = predict_fruit(saved_path)
            image_path = f"uploads/{filename}"

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        image_path=image_path,
        error=error,
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="127.0.0.1", port=port, debug=True)

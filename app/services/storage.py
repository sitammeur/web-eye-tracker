import os
from pathlib import Path
from flask import Flask
from werkzeug.utils import secure_filename


UPLOAD_FOLDER = f"{Path().absolute()}/public/videos"

# Initialize Flask app
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


def save_file_locally(file, folder):
    """
    Save a file locally in the specified folder.

    Args:
        file: The file to be saved.
        folder: The folder where the file will be saved.

    Returns:
        The path of the saved file relative to the specified folder.
    """
    # Create folder if does not exists
    os.makedirs(UPLOAD_FOLDER + folder, exist_ok=True)

    # Save file
    filename = secure_filename(file.filename)
    file.save(os.path.join(app.config["UPLOAD_FOLDER"] + folder, filename))

    # Return file path
    return f"{folder}/{filename}"

import io
import json
import logging
import os
import tempfile

import numpy as np
import torch
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure Hugging Face cache is set to a persistent location
# This prevents re-downloading models on every restart
cache_dir = os.path.expanduser("~/.cache/huggingface")
os.makedirs(cache_dir, exist_ok=True)
os.environ.setdefault("HF_HOME", cache_dir)
os.environ.setdefault("TRANSFORMERS_CACHE", cache_dir)

app = FastAPI()

# Initialize the OCR model
# Check if CUDA is available, otherwise use CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

logger.info(f"Loading OCR model (device: {device})...")
logger.info(f"Using cache directory: {cache_dir}")

predictor = ocr_predictor(
    pretrained=True,
    det_arch="fast_base",
    reco_arch="parseq",
    assume_straight_pages=False,
    detect_orientation=True,
    disable_crop_orientation=False,
    disable_page_orientation=False,
    straighten_pages=True,
)

logger.info("OCR model loaded successfully")

# Move to device and set precision
if device == "cuda":
    predictor = predictor.cuda().half()
else:
    predictor = predictor.eval()


def convert_to_json_serializable(obj):
    """Convert numpy types and other non-serializable types to JSON-serializable."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_json_serializable(item) for item in obj)
    else:
        return obj


@app.get("/")
def read_root():
    return {"message": "OCR API - Upload an image to /ocr endpoint"}


@app.get("/health")
def health_check():
    return {"status": "healthy", "device": device}


@app.post("/ocr")
async def ocr_image(file: UploadFile = File(...)):
    """
    OCR endpoint that accepts an image file and returns OCR results as JSON.

    Args:
        file: Image file (JPEG, PNG, etc.)

    Returns:
        JSON response with OCR results
    """
    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        # Read the uploaded file
        contents = await file.read()

        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            tmp_file.write(contents)
            tmp_path = tmp_file.name

        try:
            # Create DocumentFile from file path
            doc = DocumentFile.from_images([tmp_path])

            # Run OCR
            result = predictor(doc)

            # Export to JSON
            json_export = result.export()

            # Convert numpy types to JSON-serializable types
            json_export = convert_to_json_serializable(json_export)

            return JSONResponse(content=json_export)
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    except Exception as e:
        error_msg = f"Error processing image: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(status_code=500, detail=error_msg)

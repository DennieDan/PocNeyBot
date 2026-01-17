import json
import logging
import os
import tempfile
import uuid
from datetime import datetime
from typing import Optional, Union

import numpy as np
import torch
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator
from pypdf import PdfReader

from tools import call_llm
from tools.llm_system_prompt import ANNOYING_PROMPT, SYSTEM_PROMPT

# Load environment variables from .env file
load_dotenv()

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
    """Convert numpy types to JSON-serializable."""
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
    return {
        "message": "OCR API - Upload an image or PDF to /ocr endpoint",
        "supported_formats": [
            "image/*",
            "application/pdf",
            "application/pdfa",
        ],
    }


@app.get("/health")
def health_check():
    return {"status": "healthy", "device": device}


def extract_text_from_pdf(pdf_path: str) -> Optional[str]:
    """
    Extract text from PDF if it has text layers.

    Args:
        pdf_path: Path to PDF file

    Returns:
        Extracted text if PDF has text layers, None otherwise
    """
    try:
        reader = PdfReader(pdf_path)
        text_parts = []

        for page in reader.pages:
            page_text = page.extract_text()
            if page_text and page_text.strip():
                text_parts.append(page_text)

        if text_parts:
            return "\n".join(text_parts)
        return None
    except Exception as e:
        logger.warning(f"Could not extract text from PDF: {str(e)}")
        return None


def create_ocr_result_from_text(text: str) -> dict:
    """
    Create a doctr-like OCR result structure from plain text.

    This allows using the same prompt processing for both OCR and
    text extraction.

    Args:
        text: Plain text content

    Returns:
        Dictionary in doctr OCR result format
    """
    lines = text.split("\n")
    words_list = []

    for line_idx, line in enumerate(lines):
        if not line.strip():
            continue

        words = line.split()
        line_words = []
        for word_idx, word in enumerate(words):
            # Create a simple geometry (normalized coordinates)
            # This is a simplified representation
            line_words.append(
                {
                    "value": word,
                    "confidence": 1.0,
                    "geometry": [
                        [word_idx * 0.1, line_idx * 0.05],
                        [(word_idx + 1) * 0.1, line_idx * 0.05],
                        [(word_idx + 1) * 0.1, (line_idx + 1) * 0.05],
                        [word_idx * 0.1, (line_idx + 1) * 0.05],
                    ],
                    "objectness_score": 1.0,
                    "crop_orientation": {"value": 0, "confidence": 1.0},
                }
            )

        if line_words:
            words_list.append(
                {
                    "geometry": [
                        [0.0, line_idx * 0.05],
                        [1.0, line_idx * 0.05],
                        [1.0, (line_idx + 1) * 0.05],
                        [0.0, (line_idx + 1) * 0.05],
                    ],
                    "objectness_score": 1.0,
                    "words": line_words,
                }
            )

    return {
        "pages": [
            {
                "page_idx": 0,
                "dimensions": [960, 540],
                "orientation": {"value": 0, "confidence": None},
                "language": {"value": None, "confidence": None},
                "blocks": [
                    {
                        "geometry": [
                            [0.0, 0.0],
                            [1.0, 0.0],
                            [1.0, 1.0],
                            [0.0, 1.0],
                        ],
                        "objectness_score": 1.0,
                        "lines": words_list,
                        "artefacts": [],
                    }
                ],
            }
        ]
    }


@app.post("/ocr_render")
async def ocr_render(file: UploadFile = File(...)):
    """
    OCR render endpoint that accepts an image or PDF file and returns a rendered PDF file.

    Args:
        file: Image file (JPEG, PNG, etc.) or PDF file (including PDF/A)

    Returns:
        Texts in same lines
    """
    # Validate file type
    is_pdf = file.content_type and file.content_type in [
        "application/pdf",
        "application/pdfa",
    ]
    is_image = file.content_type and file.content_type.startswith("image/")

    if not (is_pdf or is_image):
        raise HTTPException(
            status_code=400,
            detail="File must be an image (JPEG, PNG, etc.) or PDF",
        )

    try:
        # Read the uploaded file
        contents = await file.read()

        # Determine file extension
        if is_pdf:
            suffix = ".pdf"
        else:
            # Try to determine from content type
            suffix = ".jpg"
            if file.content_type == "image/png":
                suffix = ".png"

        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(contents)
            tmp_path = tmp_file.name

        try:
            # For PDFs, try to extract text first (if it has text layers)
            if is_pdf:
                extracted_text = extract_text_from_pdf(tmp_path)

                if extracted_text:
                    # PDF has text layers - use text extraction
                    # (faster, more accurate)
                    logger.info("PDF has text layers, using text extraction")
                    json_export = create_ocr_result_from_text(extracted_text)
                    json_export = convert_to_json_serializable(json_export)
                    return JSONResponse(
                        content={
                            "result": json_export,
                            "method": "text_extraction",
                            "note": "PDF had text layers, no OCR needed",
                        }
                    )
                else:
                    # Scanned PDF - need OCR
                    logger.info("PDF appears to be scanned, using OCR")
                    doc = DocumentFile.from_pdf(tmp_path)
            else:
                # Image file - use OCR
                doc = DocumentFile.from_images([tmp_path])

            # Run OCR
            result = predictor(doc)

            # Extract text from the OCR result
            text = result.render()

            # Call LLM
            prompt = SYSTEM_PROMPT + "\n\n" + text
            response = call_llm.call_llm_json(prompt)

            return JSONResponse(content=response)
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    except Exception as e:
        error_msg = f"Error processing file: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(status_code=500, detail=error_msg)


@app.post("/ocr")
async def ocr_image(file: UploadFile = File(...)):
    """
    OCR endpoint that accepts an image or PDF file and returns OCR results.

    For PDFs with text layers, extracts text directly (faster, more accurate).
    For scanned PDFs or images, performs OCR.

    Args:
        file: Image file (JPEG, PNG, etc.) or PDF file (including PDF/A)

    Returns:
        JSON response with OCR results in doctr format
    """
    # Validate file type
    is_pdf = file.content_type and file.content_type in [
        "application/pdf",
        "application/pdfa",
    ]
    is_image = file.content_type and file.content_type.startswith("image/")

    if not (is_pdf or is_image):
        raise HTTPException(
            status_code=400,
            detail="File must be an image (JPEG, PNG, etc.) or PDF",
        )

    try:
        # Read the uploaded file
        contents = await file.read()

        # Determine file extension
        if is_pdf:
            suffix = ".pdf"
        else:
            # Try to determine from content type
            suffix = ".jpg"
            if file.content_type == "image/png":
                suffix = ".png"

        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(contents)
            tmp_path = tmp_file.name

        try:
            # For PDFs, try to extract text first (if it has text layers)
            if is_pdf:
                extracted_text = extract_text_from_pdf(tmp_path)

                if extracted_text:
                    # PDF has text layers - use text extraction
                    # (faster, more accurate)
                    logger.info("PDF has text layers, using text extraction")
                    json_export = create_ocr_result_from_text(extracted_text)
                    json_export = convert_to_json_serializable(json_export)
                    return JSONResponse(
                        content={
                            "result": json_export,
                            "method": "text_extraction",
                            "note": "PDF had text layers, no OCR needed",
                        }
                    )
                else:
                    # Scanned PDF - need OCR
                    logger.info("PDF appears to be scanned, using OCR")
                    doc = DocumentFile.from_pdf(tmp_path)
            else:
                # Image file - use OCR
                doc = DocumentFile.from_images([tmp_path])

            # Run OCR
            result = predictor(doc)

            # Export to JSON
            json_export = result.export()

            # Convert numpy types to JSON-serializable types
            json_export = convert_to_json_serializable(json_export)

            return JSONResponse(
                content={
                    "result": json_export,
                    "method": "ocr",
                    "note": "OCR was performed on image/scanned PDF",
                }
            )
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    except Exception as e:
        error_msg = f"Error processing file: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(status_code=500, detail=error_msg)


# Pydantic models for manual input endpoint
class ManualInputRequest(BaseModel):
    amount: float
    note: str
    category: Optional[int] = None
    method: Optional[int] = None
    occurredAt: Optional[datetime] = None
    currency: Optional[str] = "SGD"
    merchant: Optional[str] = None


class ManualInputResponse(BaseModel):
    id: str
    method: str
    aiComment: str
    category: str
    amount: float
    currency: str
    item: Optional[str] = None
    merchant: Optional[str] = None
    occurredAt: datetime
    note: Optional[str] = None


class OcrRequest(BaseModel):
    file: UploadFile = File(...)


class OcrItem(BaseModel):
    item: str
    price: float
    quantity: int
    total: float


class OcrItemsResponse(BaseModel):
    items: list[OcrItem]


class TransactionInput(BaseModel):
    method: str
    aiComment: str = ""
    category: str
    item: str
    amount: Union[str, float]
    currency: str
    merchant: str
    occurredAt: str
    note: str

    @field_validator("amount", mode="before")
    @classmethod
    def convert_amount(cls, v):
        """Convert string amount to float if needed."""
        if isinstance(v, str):
            return float(v)
        return v


@app.post("/manual-input", response_model=ManualInputResponse)
async def manual_input(data: ManualInputRequest):
    """
    Manual input endpoint that accepts transaction data.

    Args:
        data: Transaction input data containing amount, note,
              and optional fields

    Returns:
        JSON response with transaction details including id,
        payment_method, ai_comment, category, and amount
    """
    # Generate a unique transaction ID
    transaction_id = str(uuid.uuid4())

    # Placeholder logic for payment_method (can be enhanced later)
    # For now, using a simple mapping or default value
    payment_method = "Cash"

    # Placeholder logic for ai_comment (can be enhanced with AI later)
    ai_comment = f"Transaction processed: {data.note}"

    currency = "SGD"

    merchant = "Unknown"

    # Placeholder logic for category (convert from number to string)
    # For now, using a simple mapping
    category_map = {
        1: "Food",
        2: "Transport",
        3: "Shopping",
        4: "Bills",
        5: "Entertainment",
        6: "Other",
    }
    if data.category:
        category_str = category_map.get(data.category, "Uncategorized")
    else:
        category_str = "Uncategorized"

    response = ManualInputResponse(
        id=transaction_id,
        method=payment_method,
        aiComment=ai_comment,
        category=category_str,
        amount=data.amount,
        currency=currency,
        item=None,
        merchant=merchant,
        occurredAt=datetime.now(),
        note=data.note,
    )

    return response


@app.post("/call_ai_comment")
async def call_ai_comment(data: list[TransactionInput]):
    """
    Call AI comment endpoint that accepts transaction data.

    Args:
        data: List of transaction input data containing amount, note,
              and optional fields

    Returns:
        JSON response with AI comment
    """
    json_input = [item.model_dump() for item in data]
    prompt = ANNOYING_PROMPT + "\n\n" + json.dumps(json_input)
    ai_comment = {"ai_comment": call_llm.call_llm(prompt), "status": 200}

    return JSONResponse(content=ai_comment)

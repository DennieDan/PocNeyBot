import base64
import os
from tempfile import TemporaryDirectory
from typing import List, Union

import numpy as np
from ocrmypdf.hocrtransform import HocrTransform
from PIL import Image


def convert_to_pdf(
    result, docs: List[np.ndarray], output_dir: str = None, save_base64: bool = False
) -> Union[str, List[str], List[bytes]]:
    """
    Convert OCR result to PDF.

    Args:
        result: OCR result object from doctr (with export_as_xml method)
        docs: List of document images as numpy arrays
        output_dir: Directory to save PDFs. If None, uses temporary
            directory.
        save_base64: If True, returns base64-encoded PDFs. If False,
            returns file paths.

    Returns:
        If save_base64 is False: List of PDF file paths (or single path
            if one page)
        If save_base64 is True: List of base64-encoded PDF bytes
    """
    # Get XML outputs from OCR result
    xml_outputs = result.export_as_xml()

    # Prepare output
    pdf_paths = []
    base64_encoded_pdfs = []

    # Determine if we should use a temporary directory or provided
    # directory
    # If saving as base64, we can use temp directory
    # If saving as files, we need a persistent directory
    if save_base64 or output_dir is not None:
        if output_dir is None:
            # Use temporary directory for base64 encoding
            with TemporaryDirectory() as work_dir:
                return _process_pages(
                    xml_outputs,
                    docs,
                    work_dir,
                    pdf_paths,
                    base64_encoded_pdfs,
                    save_base64,
                )
        else:
            # Use provided directory
            os.makedirs(output_dir, exist_ok=True)
            return _process_pages(
                xml_outputs,
                docs,
                output_dir,
                pdf_paths,
                base64_encoded_pdfs,
                save_base64,
            )
    else:
        # Need output_dir if not saving as base64
        raise ValueError("output_dir must be provided when save_base64 is False")


def _process_pages(
    xml_outputs,
    docs: List[np.ndarray],
    work_dir: str,
    pdf_paths: List[str],
    base64_encoded_pdfs: List[bytes],
    save_base64: bool,
) -> Union[str, List[str], List[bytes]]:
    """Process pages and convert to PDF."""
    # Process each page
    for i, (xml, img) in enumerate(zip(xml_outputs, docs)):
        # Write the image temporarily
        img_path = os.path.join(work_dir, f"{i}.jpg")
        Image.fromarray(img).save(img_path)

        # Write the XML content temporarily
        xml_path = os.path.join(work_dir, f"{i}.xml")
        with open(xml_path, "w", encoding="utf-8") as f:
            f.write(xml_outputs[i][0].decode())

        # Initialize hOCR transformer
        hocr = HocrTransform(hocr_filename=xml_path, dpi=300)

        # Save as PDF/A
        pdf_path = os.path.join(work_dir, f"{i}.pdf")
        hocr.to_pdf(out_filename=pdf_path, image_filename=img_path)

        if save_base64:
            # Read and encode PDF to base64
            with open(pdf_path, "rb") as f:
                base64_encoded_pdfs.append(base64.b64encode(f.read()))
        else:
            pdf_paths.append(pdf_path)

    if save_base64:
        print(f"{len(base64_encoded_pdfs)} PDFs encoded")
        return base64_encoded_pdfs
    else:
        # If only one PDF, return single path; otherwise return list
        if len(pdf_paths) == 1:
            return pdf_paths[0]
        return pdf_paths

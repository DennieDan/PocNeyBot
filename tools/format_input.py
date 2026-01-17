class InputLLM(BaseModel):
    text: str
    avg_coordinates: list[float]


def format_input_llm(ocr_result: dict) -> list[InputLLM]:
    """
    Format the OCR result into a list of items.

    Args:
        ocr_result: The OCR result

    Returns:
        The list of items with the avg coordinators
    """
    response = []
    pages = ocr_result["result"]["pages"]
    for page in pages:
        for block in page["blocks"]:
            for line in block["lines"]:
                for word in line["words"]:
                    response.append(
                        InputLLM(text=word["value"], avg_coordinates=word["geometry"])
                    )
    return ocr_result

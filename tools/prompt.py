"""
Prompt template for extracting structured transaction data from doctr OCR.
"""

RECEIPT_EXTRACTION_PROMPT = """You are an expert at extracting structured transaction data from OCR results of receipts.

Your task is to analyze the OCR output from a receipt and extract all purchased items with their details.

**Input Format:**
The OCR output is a JSON structure containing:
- `pages`: Array of pages, each containing `blocks`
- `blocks`: Array of text blocks, each containing `lines`
- `lines`: Array of text lines, each containing `words`
- `words`: Individual words with their `value` (text content) and `geometry` (position)

**Output Format:**
You must return a valid JSON array where each object represents a purchased item with the following fields:
- `item`: string - The full name/description of the item
- `price`: number - The unit price of the item (as a number, not a string)
- `quantity`: number - The quantity purchased (default to 1 if not specified)
- `total`: number - The total price for this item (price × quantity, or the line total if available)

**Instructions:**
1. Look for item names in the receipt. These are typically in the left/middle portion.
2. Look for prices in the right portion. Prices are usually aligned to the right.
3. Match each item with its corresponding price based on their vertical position (y-coordinate in geometry).
4. If quantity is not explicitly stated, assume quantity = 1.
5. Calculate total as: total = price × quantity (unless a specific total is shown for that line).
6. Ignore header information (store name, address, phone, etc.), footer text, and summary sections (TOTAL, CASH, CHANGE, GST, etc.).
7. Only extract actual purchased items from the itemized list.
8. Remove currency symbols ($, SGD, etc.) from price values - store as numbers only.
9. If an item appears multiple times, create separate entries for each occurrence.

**Example Output:**
```json
[
    {
        "item": "COLLAR SHABU SHABU",
        "price": 7.70,
        "quantity": 1,
        "total": 7.70
    },
    {
        "item": "PSR AUST WNGBOK",
        "price": 1.77,
        "quantity": 1,
        "total": 1.77
    },
    {
        "item": "PSR SEA CUCUMBER",
        "price": 6.62,
        "quantity": 1,
        "total": 6.62
    }
]
```

**Important:**
- Return ONLY valid JSON, no additional text or explanation
- Ensure all numbers are actual numbers, not strings
- If you cannot determine a price or item name, skip that item
- Be precise with matching items to their prices based on line position

Now, analyze the following OCR output and extract the transaction items:
"""


def get_extraction_prompt(ocr_output: dict) -> str:
    """
    Generate a complete prompt for OpenAI to extract transaction data.

    Args:
        ocr_output: The doctr OCR output as a dictionary.
                    Can be direct OCR result or API response with "result" key.

    Returns:
        Complete prompt string ready to send to OpenAI
    """
    import json

    # Handle both direct OCR output and API response format
    if "result" in ocr_output:
        ocr_data = ocr_output["result"]
    else:
        ocr_data = ocr_output

    # Convert OCR output to JSON string
    ocr_json = json.dumps(ocr_data, indent=2)

    # Combine prompt template with OCR data
    full_prompt = f"{RECEIPT_EXTRACTION_PROMPT}\n\n```json\n{ocr_json}\n```"

    return full_prompt


def extract_text_from_ocr(ocr_output: dict) -> str:
    """
    Extract a simplified text representation from OCR output.

    Args:
        ocr_output: The doctr OCR output as a dictionary

    Returns:
        Plain text representation of the receipt
    """
    text_lines = []

    for page in ocr_output.get("pages", []):
        for block in page.get("blocks", []):
            for line in block.get("lines", []):
                words = [word.get("value", "") for word in line.get("words", [])]
                if words:
                    text_lines.append(" ".join(words))

    return "\n".join(text_lines)


# Alternative simpler prompt for text-based extraction
SIMPLE_EXTRACTION_PROMPT = """Extract all purchased items from this receipt text and return them as a JSON array.

Each item should have:
- "item": item name/description
- "price": unit price (number)
- "quantity": quantity purchased (number, default 1)
- "total": total price for this item (number)

Return ONLY valid JSON array, no other text.

Receipt text:
{receipt_text}
"""

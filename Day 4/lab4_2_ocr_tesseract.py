# =============================================================================
# LAB 4.2: OCR Pipeline with Tesseract
# Module 3 | Chitkara University | B.Tech AI Specialization
# Theme: Extract text from document images, apply preprocessing to improve
#        accuracy, and use regex to extract structured fields (date, amount, email).
# =============================================================================

# ── INSTALL (run once in terminal) ───────────────────────────────────────────
# pip install pytesseract opencv-python Pillow
#
# System-level Tesseract install (required):
#   Windows : https://github.com/UB-Mannheim/tesseract/wiki
#             Then set: pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
#   Mac     : brew install tesseract
#   Linux   : sudo apt install tesseract-ocr

import cv2
import re
import json
import os
import numpy as np
import pytesseract
from PIL import Image

# ── For Windows only — set Tesseract path ────────────────────────────────────
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Verify Tesseract is installed
try:
    version = pytesseract.get_tesseract_version()
    print(f"[INFO] Tesseract version: {version}")
except Exception:
    print("[ERROR] Tesseract not found. Check installation and PATH.")
    print("        Visit: https://github.com/UB-Mannheim/tesseract/wiki")


# =============================================================================
# SECTION 1: BASIC OCR — Extract all text from a document image
# =============================================================================

def basic_ocr(image_path: str) -> str:
    """
    Runs basic Tesseract OCR on a raw image with no preprocessing.
    Returns extracted text as a string.
    """
    print(f"\n[OCR - Basic] Processing: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        print(f"  [ERROR] Could not load image: {image_path}")
        return ""

    # Convert BGR (OpenCV) to RGB (Tesseract expects RGB)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    # Run OCR
    # config: --psm 6 = assume a single uniform block of text (good for docs)
    text = pytesseract.image_to_string(pil_img, config='--psm 6')
    print(f"  Extracted {len(text)} characters")
    return text


# =============================================================================
# SECTION 2: PREPROCESSING — Improve OCR accuracy before running Tesseract
# =============================================================================

def preprocess_for_ocr(image_path: str, save_debug: bool = True) -> np.ndarray:
    """
    Applies a multi-step preprocessing pipeline to improve OCR accuracy:
    1. Grayscale conversion
    2. Denoising
    3. Adaptive thresholding (binarisation)
    4. Deskewing (rotation correction)
    5. Upscaling (if low resolution)

    Returns the preprocessed image as a numpy array.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Could not load: {image_path}")
        return None

    print(f"\n[PREPROCESS] {image_path}  (original size: {img.shape[1]}×{img.shape[0]})")

    # ── Step 1: Grayscale ─────────────────────────────────────────────────────
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print("  ✓ Step 1: Grayscale conversion")

    # ── Step 2: Denoise (removes salt-and-pepper noise from scanned docs) ─────
    denoised = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)
    print("  ✓ Step 2: Denoising")

    # ── Step 3: Adaptive Thresholding (binarisation) ──────────────────────────
    # Better than global threshold for uneven lighting (scanned docs, shadows)
    binary = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # Uses Gaussian-weighted neighbourhood
        cv2.THRESH_BINARY,
        blockSize=11,   # Neighbourhood size (must be odd)
        C=2             # Constant subtracted from mean
    )
    print("  ✓ Step 3: Adaptive thresholding")

    # ── Step 4: Deskew ────────────────────────────────────────────────────────
    deskewed = deskew_image(binary)
    print("  ✓ Step 4: Deskewing")

    # ── Step 5: Upscale if resolution is too low (< 300 DPI equivalent) ───────
    h, w = deskewed.shape[:2]
    if w < 1000:
        scale = 2.0
        deskewed = cv2.resize(deskewed, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_CUBIC)
        print(f"  ✓ Step 5: Upscaled 2× (new size: {deskewed.shape[1]}×{deskewed.shape[0]})")
    else:
        print(f"  ✓ Step 5: No upscaling needed ({w}px wide)")

    # ── Save debug image ──────────────────────────────────────────────────────
    if save_debug:
        base = os.path.splitext(os.path.basename(image_path))[0]
        debug_path = f"{base}_preprocessed.jpg"
        cv2.imwrite(debug_path, deskewed)
        print(f"  → Preprocessed image saved: {debug_path}")

    return deskewed


def deskew_image(gray_img: np.ndarray) -> np.ndarray:
    """
    Detects and corrects text skew using Hough line transform.
    Returns the straightened image.
    """
    # Find edges
    edges = cv2.Canny(gray_img, 50, 150, apertureSize=3)

    # Detect lines
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                             threshold=100, minLineLength=100, maxLineGap=10)
    if lines is None:
        return gray_img  # No lines detected, return original

    # Calculate average angle
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 - x1 != 0:  # Avoid division by zero
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if abs(angle) < 45:  # Only near-horizontal lines
                angles.append(angle)

    if not angles:
        return gray_img

    median_angle = np.median(angles)
    if abs(median_angle) < 0.5:
        return gray_img  # Already straight

    # Rotate to correct skew
    h, w = gray_img.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    rotated = cv2.warpAffine(gray_img, rotation_matrix, (w, h),
                              flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)
    return rotated


# =============================================================================
# SECTION 3: COMPARE ACCURACY — Raw vs Preprocessed OCR
# =============================================================================

def compare_ocr_accuracy(image_path: str, ground_truth_text: str = None):
    """
    Runs OCR on both raw and preprocessed versions of an image.
    Compares character count and optionally measures accuracy against ground truth.
    """
    print(f"\n{'='*60}")
    print(f"COMPARING OCR: Raw vs Preprocessed")
    print(f"{'='*60}")

    # Raw OCR
    raw_text = basic_ocr(image_path)

    # Preprocessed OCR
    preprocessed_img = preprocess_for_ocr(image_path, save_debug=True)
    if preprocessed_img is None:
        return

    pil_preprocessed = Image.fromarray(preprocessed_img)
    preprocessed_text = pytesseract.image_to_string(pil_preprocessed, config='--psm 6')

    # Report
    print(f"\n── RESULTS ─────────────────────────────────────────────")
    print(f"  Raw OCR      : {len(raw_text.strip())} characters extracted")
    print(f"  Preprocessed : {len(preprocessed_text.strip())} characters extracted")

    if ground_truth_text:
        def char_accuracy(extracted, truth):
            extracted_clean = ''.join(extracted.split())
            truth_clean = ''.join(truth.split())
            if not truth_clean:
                return 0
            matches = sum(1 for a, b in zip(extracted_clean, truth_clean) if a == b)
            return matches / len(truth_clean) * 100

        raw_acc = char_accuracy(raw_text, ground_truth_text)
        pre_acc = char_accuracy(preprocessed_text, ground_truth_text)
        improvement = pre_acc - raw_acc
        print(f"\n  Raw accuracy        : {raw_acc:.1f}%")
        print(f"  Preprocessed accur  : {pre_acc:.1f}%")
        print(f"  Improvement         : +{improvement:.1f}%")

    return raw_text, preprocessed_text


# =============================================================================
# SECTION 4: REGEX FIELD EXTRACTION — Dates, amounts, emails
# =============================================================================

def extract_invoice_fields(text: str) -> dict:
    """
    Uses regular expressions to extract common invoice fields from OCR text.
    Returns a structured dictionary of extracted fields.
    """
    print("\n[EXTRACT] Running regex field extraction...")

    fields = {}

    # ── Dates ──────────────────────────────────────────────────────────────────
    date_patterns = [
        r'\b\d{2}/\d{2}/\d{4}\b',           # DD/MM/YYYY or MM/DD/YYYY
        r'\b\d{4}-\d{2}-\d{2}\b',            # YYYY-MM-DD (ISO format)
        r'\b\d{1,2}\s+\w{3,9}\s+\d{4}\b',   # 15 January 2024
        r'\b\w{3,9}\s+\d{1,2},?\s+\d{4}\b', # January 15, 2024
    ]
    all_dates = []
    for pattern in date_patterns:
        all_dates.extend(re.findall(pattern, text, re.IGNORECASE))
    fields["dates"] = list(set(all_dates))

    # ── Currency amounts ──────────────────────────────────────────────────────
    amount_patterns = [
        r'\$[\d,]+\.?\d{0,2}',              # $1,234.56 or $1234
        r'₹[\d,]+\.?\d{0,2}',              # ₹1,234.56 (Indian Rupee)
        r'Rs\.?\s*[\d,]+\.?\d{0,2}',        # Rs. 1234
        r'\b\d{1,3}(?:,\d{3})*(?:\.\d{2})?\b(?=\s*(?:USD|INR|EUR))',  # 1,234.56 USD
    ]
    all_amounts = []
    for pattern in amount_patterns:
        all_amounts.extend(re.findall(pattern, text))
    fields["amounts"] = list(set(all_amounts))

    # ── Total amount (labelled) ───────────────────────────────────────────────
    total_pattern = r'(?:total|amount due|grand total|balance due)[:\s]*[\$₹Rs\.]*\s*([\d,]+\.?\d{0,2})'
    total_match = re.search(total_pattern, text, re.IGNORECASE)
    fields["total"] = total_match.group(1).replace(',', '') if total_match else None

    # ── Invoice number ────────────────────────────────────────────────────────
    inv_pattern = r'(?:invoice\s*(?:no|number|#)[:\s]*)([\w\-]+)'
    inv_match = re.search(inv_pattern, text, re.IGNORECASE)
    fields["invoice_number"] = inv_match.group(1) if inv_match else None

    # ── Email addresses ───────────────────────────────────────────────────────
    email_pattern = r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}'
    fields["emails"] = re.findall(email_pattern, text)

    # ── Phone numbers ─────────────────────────────────────────────────────────
    phone_pattern = r'(?:\+?\d{1,3}[\s\-]?)?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{4}'
    fields["phones"] = re.findall(phone_pattern, text)

    # ── Vendor / Company Name (first capitalised multi-word line) ─────────────
    vendor_pattern = r'^([A-Z][A-Za-z\s&.,]{5,50})$'
    vendor_matches = re.findall(vendor_pattern, text, re.MULTILINE)
    fields["possible_vendor"] = vendor_matches[0].strip() if vendor_matches else None

    # Print summary
    for key, value in fields.items():
        print(f"  {key:20s}: {value}")

    return fields


# =============================================================================
# SECTION 5: FULL INVOICE PIPELINE — End-to-end for one image
# =============================================================================

def process_invoice(image_path: str) -> dict:
    """
    Full pipeline: Load → Preprocess → OCR → Extract Fields → JSON output
    Returns structured invoice data dictionary.
    """
    print(f"\n{'='*60}")
    print(f"PROCESSING INVOICE: {image_path}")
    print(f"{'='*60}")

    # 1. Preprocess the image
    preprocessed = preprocess_for_ocr(image_path, save_debug=True)
    if preprocessed is None:
        return {"error": f"Could not load: {image_path}"}

    # 2. Run OCR
    pil_img = Image.fromarray(preprocessed)
    raw_text = pytesseract.image_to_string(pil_img, config='--psm 6 --oem 3')
    print(f"\n[OCR] Extracted text ({len(raw_text)} chars):")
    print("-" * 40)
    print(raw_text[:500] + ("..." if len(raw_text) > 500 else ""))
    print("-" * 40)

    # 3. Extract structured fields
    fields = extract_invoice_fields(raw_text)

    # 4. Structure as invoice object
    invoice_data = {
        "source_file": os.path.basename(image_path),
        "raw_text": raw_text,
        "extracted_fields": fields,
        "validation": {
            "has_date": bool(fields.get("dates")),
            "has_amount": bool(fields.get("amounts") or fields.get("total")),
            "has_invoice_number": bool(fields.get("invoice_number")),
            "fields_found": sum([
                bool(fields.get("dates")),
                bool(fields.get("amounts")),
                bool(fields.get("invoice_number")),
                bool(fields.get("emails"))
            ])
        }
    }

    return invoice_data


# =============================================================================
# SECTION 6: BATCH PROCESSING — Run pipeline on a folder of invoices
# =============================================================================

def process_invoice_batch(folder_path: str, output_file: str = "invoices_output.json"):
    """
    Processes all invoice images in a folder and saves results as JSON.
    Matches the Mini Project requirement of 50 invoices.
    """
    if not os.path.exists(folder_path):
        print(f"[ERROR] Folder not found: {folder_path}")
        return

    supported_ext = ('.jpg', '.jpeg', '.png', '.tiff', '.bmp')
    image_files = [f for f in os.listdir(folder_path)
                   if f.lower().endswith(supported_ext)]
    image_files.sort()

    print(f"\n[BATCH] Found {len(image_files)} images in {folder_path}")
    print(f"[BATCH] Processing all invoices...\n")

    all_results = []
    success_count = 0

    for i, filename in enumerate(image_files, 1):
        img_path = os.path.join(folder_path, filename)
        print(f"\n[{i:2d}/{len(image_files)}] {filename}")

        try:
            result = process_invoice(img_path)
            all_results.append(result)
            if not result.get("error"):
                success_count += 1
        except Exception as e:
            print(f"  [ERROR] Failed: {e}")
            all_results.append({"source_file": filename, "error": str(e)})

    # Save all results as JSON
    output = {
        "total_invoices": len(image_files),
        "successfully_processed": success_count,
        "invoices": all_results
    }
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*60}")
    print(f"[BATCH COMPLETE]")
    print(f"  Processed : {success_count}/{len(image_files)} invoices")
    print(f"  Output    : {output_file}")
    print(f"{'='*60}")

    return output


# =============================================================================
# MAIN — Run the full lab workflow
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  LAB 4.2 — OCR Pipeline with Tesseract")
    print("  Module 3 | Chitkara University")
    print("=" * 60)

    # ── Create a sample test image if no invoice is available ────────────────
    # This generates a simple text image to test OCR without needing a real invoice
    def create_sample_invoice():
        img = np.ones((600, 800, 3), dtype=np.uint8) * 255  # White background
        lines = [
            ("ACME SUPPLIES LTD", (80, 60), 1.0),
            ("Invoice No: INV-2024-00123", (80, 120), 0.7),
            ("Date: 15/02/2024", (80, 170), 0.7),
            ("Bill To: customer@example.com", (80, 220), 0.7),
            ("Phone: +91-98765-43210", (80, 270), 0.7),
            ("", (0, 0), 0),
            ("Item 1 - Widget A         Rs. 1,200.00", (80, 360), 0.65),
            ("Item 2 - Widget B         Rs. 800.00", (80, 400), 0.65),
            ("Item 3 - Shipping         Rs. 150.00", (80, 440), 0.65),
            ("", (0, 0), 0),
            ("TOTAL DUE: Rs. 2,150.00", (80, 520), 0.9),
        ]
        for text, pos, scale in lines:
            if text:
                cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                            scale, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.imwrite("sample_invoice.jpg", img)
        print("[INFO] Created sample_invoice.jpg for testing")

    create_sample_invoice()

    # ── STEP 1: Basic OCR on a raw image ─────────────────────────────────────
    print("\n[STEP 1] Basic OCR (no preprocessing)")
    basic_text = basic_ocr("sample_invoice.jpg")
    print("Extracted text:\n", basic_text)

    # ── STEP 2: OCR with preprocessing ───────────────────────────────────────
    print("\n[STEP 2] OCR with full preprocessing pipeline")
    preprocessed = preprocess_for_ocr("sample_invoice.jpg")
    if preprocessed is not None:
        pil_img = Image.fromarray(preprocessed)
        clean_text = pytesseract.image_to_string(pil_img, config='--psm 6')
        print("Extracted text:\n", clean_text)

    # ── STEP 3: Extract fields with regex ────────────────────────────────────
    print("\n[STEP 3] Regex field extraction")
    sample_text = """
    ACME SUPPLIES LTD
    Invoice No: INV-2024-00123
    Date: 15/02/2024
    customer@example.com
    Phone: +91-98765-43210
    Item 1 - Widget A    Rs. 1,200.00
    Item 2 - Widget B    Rs. 800.00
    Item 3 - Shipping    Rs. 150.00
    TOTAL: Rs. 2,150.00
    """
    fields = extract_invoice_fields(sample_text)

    # ── STEP 4: Full invoice pipeline ────────────────────────────────────────
    print("\n[STEP 4] Full invoice pipeline")
    result = process_invoice("sample_invoice.jpg")
    with open("invoice_result.json", "w") as f:
        json.dump(result, f, indent=2)
    print("\n[INFO] Result saved to: invoice_result.json")

    # ── STEP 5: Batch processing (for Mini Project) ───────────────────────────
    # Uncomment when you have your invoice folder ready:
    # process_invoice_batch("./invoices/", "all_invoices.json")

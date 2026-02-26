# =============================================================================
# LAB 5.1: Azure Computer Vision API
# Module 3 | Chitkara University | B.Tech AI Specialization
# Theme: Use Azure Computer Vision to analyze product images, generate tags
#        and captions, and understand when cloud APIs beat custom models.
# =============================================================================

# ── INSTALL (run once in terminal) ───────────────────────────────────────────
# pip install azure-cognitiveservices-vision-computervision msrest requests

# ── SETUP: Get your credentials from Azure Portal ────────────────────────────
# 1. Go to https://portal.azure.com
# 2. Create a new "Computer Vision" resource (Free tier F0)
# 3. Go to resource → "Keys and Endpoint"
# 4. Copy Key 1 and Endpoint URL into the constants below

from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes, Details
from msrest.authentication import CognitiveServicesCredentials
import requests
import json
import os
import time

# =============================================================================
# CONFIGURATION — Replace with your Azure credentials
# =============================================================================

AZURE_ENDPOINT = "https://YOUR_RESOURCE_NAME.cognitiveservices.azure.com/"
AZURE_KEY      = "YOUR_API_KEY_HERE"

# ── Test images (publicly accessible URLs) ───────────────────────────────────
# These are sample product-like images you can use if you don't have your own.
SAMPLE_IMAGE_URLS = [
    "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/1/18/Dog_Breeds.jpg/1200px-Dog_Breeds.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Bikesgalore_%28cropped%29.jpg/1200px-Bikesgalore_%28cropped%29.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/0/04/Chicken_adobo.jpg/1200px-Chicken_adobo.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/Camponotus_flavomarginatus_ant.jpg/1200px-Camponotus_flavomarginatus_ant.jpg",
]


# =============================================================================
# SECTION 1: CONNECT TO AZURE
# =============================================================================

def create_client() -> ComputerVisionClient:
    """
    Creates and returns an authenticated Azure Computer Vision client.
    """
    if "YOUR_" in AZURE_ENDPOINT or "YOUR_" in AZURE_KEY:
        print("[ERROR] Replace AZURE_ENDPOINT and AZURE_KEY with your real credentials.")
        print("        Get them from: Azure Portal → Your CV Resource → Keys and Endpoint")
        return None

    client = ComputerVisionClient(
        endpoint=AZURE_ENDPOINT,
        credentials=CognitiveServicesCredentials(AZURE_KEY)
    )
    print("[INFO] Azure Computer Vision client created successfully.")
    return client


# =============================================================================
# SECTION 2: ANALYZE IMAGE — Tags, captions, objects, colours
# =============================================================================

def analyze_image_url(client: ComputerVisionClient, image_url: str,
                       confidence_threshold: float = 0.85) -> dict:
    """
    Sends an image URL to Azure Computer Vision and retrieves:
    - Tags with confidence scores
    - Descriptive captions
    - Detected objects with bounding boxes
    - Dominant colours

    Returns a structured results dictionary.
    """
    print(f"\n[ANALYZE] {image_url[:80]}...")

    # Which features to request from Azure
    features = [
        VisualFeatureTypes.tags,        # Image tags (what's in the image)
        VisualFeatureTypes.description, # Natural language captions
        VisualFeatureTypes.objects,     # Detected objects + bounding boxes
        VisualFeatureTypes.color,       # Dominant colours
        VisualFeatureTypes.categories,  # High-level image category
    ]

    try:
        # Call Azure API
        result = client.analyze_image(image_url, visual_features=features)

        # ── Extract tags above confidence threshold ───────────────────────────
        tags = []
        for tag in result.tags:
            if tag.confidence >= confidence_threshold:
                tags.append({
                    "name": tag.name,
                    "confidence": round(tag.confidence, 3)
                })

        # ── Extract captions ──────────────────────────────────────────────────
        captions = []
        if result.description and result.description.captions:
            for caption in result.description.captions:
                captions.append({
                    "text": caption.text,
                    "confidence": round(caption.confidence, 3)
                })

        # ── Extract detected objects ──────────────────────────────────────────
        objects = []
        if result.objects:
            for obj in result.objects:
                objects.append({
                    "object": obj.object_property,
                    "confidence": round(obj.confidence, 3),
                    "bounding_box": {
                        "x": obj.rectangle.x,
                        "y": obj.rectangle.y,
                        "w": obj.rectangle.w,
                        "h": obj.rectangle.h
                    }
                })

        # ── Extract colour info ───────────────────────────────────────────────
        colour_info = {}
        if result.color:
            colour_info = {
                "dominant_foreground": result.color.dominant_color_foreground,
                "dominant_background": result.color.dominant_color_background,
                "accent_color": f"#{result.color.accent_color}",
                "is_bw": result.color.is_bw_img
            }

        output = {
            "image_url": image_url,
            "tags": tags,
            "captions": captions,
            "objects": objects,
            "colors": colour_info,
            "tag_count": len(tags)
        }

        # Print summary
        print(f"  Tags ({len(tags)}): {', '.join(t['name'] for t in tags[:5])}{'...' if len(tags) > 5 else ''}")
        if captions:
            print(f"  Caption   : {captions[0]['text']}  (conf: {captions[0]['confidence']})")
        if objects:
            print(f"  Objects   : {', '.join(o['object'] for o in objects)}")

        return output

    except Exception as e:
        print(f"  [ERROR] API call failed: {e}")
        return {"image_url": image_url, "error": str(e)}


def analyze_local_image(client: ComputerVisionClient, image_path: str,
                         confidence_threshold: float = 0.85) -> dict:
    """
    Analyzes a local image file (reads and sends as stream).
    Same output structure as analyze_image_url().
    """
    print(f"\n[ANALYZE LOCAL] {image_path}")

    features = [
        VisualFeatureTypes.tags,
        VisualFeatureTypes.description,
        VisualFeatureTypes.objects,
        VisualFeatureTypes.color,
    ]

    try:
        with open(image_path, "rb") as img_file:
            result = client.analyze_image_in_stream(img_file, visual_features=features)

        tags = [{"name": t.name, "confidence": round(t.confidence, 3)}
                for t in result.tags if t.confidence >= confidence_threshold]
        captions = []
        if result.description and result.description.captions:
            captions = [{"text": c.text, "confidence": round(c.confidence, 3)}
                        for c in result.description.captions]

        output = {
            "image_path": image_path,
            "tags": tags,
            "captions": captions,
            "tag_count": len(tags)
        }

        print(f"  Tags ({len(tags)}): {', '.join(t['name'] for t in tags[:5])}")
        if captions:
            print(f"  Caption: {captions[0]['text']}")

        return output

    except Exception as e:
        print(f"  [ERROR] {e}")
        return {"image_path": image_path, "error": str(e)}


# =============================================================================
# SECTION 3: OCR WITH AZURE READ API — Extract text from images
# =============================================================================

def extract_text_from_image_url(client: ComputerVisionClient, image_url: str) -> str:
    """
    Uses Azure's Read API (async) to extract text from an image URL.
    More accurate than basic OCR for documents, handwriting, and dense text.
    Returns extracted text as a single string.
    """
    print(f"\n[READ OCR] Extracting text from: {image_url[:60]}...")

    try:
        # Start async Read operation
        read_response = client.read(url=image_url, raw=True)

        # Get operation ID from response headers
        operation_id = read_response.headers["Operation-Location"].split("/")[-1]

        # Poll until complete (usually 2–5 seconds)
        max_wait = 30  # seconds
        elapsed = 0
        while elapsed < max_wait:
            read_result = client.get_read_result(operation_id)
            if read_result.status not in ["notStarted", "running"]:
                break
            time.sleep(1)
            elapsed += 1

        # Extract text from result
        if read_result.status == "succeeded":
            extracted_lines = []
            for page in read_result.analyze_result.read_results:
                for line in page.lines:
                    extracted_lines.append(line.text)
            full_text = "\n".join(extracted_lines)
            print(f"  Extracted {len(extracted_lines)} lines of text")
            return full_text
        else:
            print(f"  [ERROR] Read operation status: {read_result.status}")
            return ""

    except Exception as e:
        print(f"  [ERROR] {e}")
        return ""


# =============================================================================
# SECTION 4: BATCH ANALYSIS — Analyze 10 images and evaluate accuracy
# =============================================================================

def run_batch_analysis(client: ComputerVisionClient,
                        image_urls: list,
                        confidence_threshold: float = 0.85,
                        output_file: str = "lab5_1_results.json") -> list:
    """
    Analyzes a list of image URLs and saves structured JSON results.
    Returns list of result dictionaries.
    """
    print(f"\n[BATCH] Analyzing {len(image_urls)} images...")
    print(f"        Confidence threshold: {confidence_threshold}")

    all_results = []

    for i, url in enumerate(image_urls, 1):
        print(f"\n[{i:2d}/{len(image_urls)}]")
        result = analyze_image_url(client, url, confidence_threshold)
        all_results.append(result)
        time.sleep(0.5)  # Respect rate limit (20 req/min on free tier)

    # Save to JSON
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n[BATCH COMPLETE] Results saved to: {output_file}")
    print(f"  Total images    : {len(all_results)}")
    print(f"  Avg tags/image  : {sum(r.get('tag_count', 0) for r in all_results) / len(all_results):.1f}")

    return all_results


# =============================================================================
# SECTION 5: MANUAL ACCURACY EVALUATION — Compare API output vs your judgment
# =============================================================================

def evaluate_tag_accuracy(results: list, num_to_evaluate: int = 3) -> dict:
    """
    For the first `num_to_evaluate` images, compares Azure tags against
    manually-assigned ground truth labels.

    Fill in `ground_truth` below with what YOU think the correct tags should be.
    """
    # ── Edit this dictionary with your manual labels ──────────────────────────
    # Format: { image_index: ["expected_tag1", "expected_tag2", ...] }
    ground_truth = {
        0: ["cat", "animal", "fur", "whiskers"],
        1: ["dog", "animal", "breed"],
        2: ["bicycle", "outdoor", "wheel"],
    }

    eval_results = []
    print(f"\n[EVAL] Manual accuracy evaluation for {num_to_evaluate} images")
    print("-" * 60)

    for idx in range(min(num_to_evaluate, len(results))):
        if idx not in ground_truth:
            continue

        api_tags = {t["name"].lower() for t in results[idx].get("tags", [])}
        expected = {t.lower() for t in ground_truth[idx]}

        true_positives = api_tags & expected
        false_positives = api_tags - expected
        false_negatives = expected - api_tags

        precision = len(true_positives) / len(api_tags) if api_tags else 0
        recall    = len(true_positives) / len(expected) if expected else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        print(f"\n  Image {idx+1}:")
        print(f"    Expected tags  : {sorted(expected)}")
        print(f"    Azure tags     : {sorted(api_tags)}")
        print(f"    Correct matches: {sorted(true_positives)}")
        print(f"    Precision      : {precision:.2f}  |  Recall: {recall:.2f}  |  F1: {f1:.2f}")

        eval_results.append({
            "image_index": idx,
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3)
        })

    avg_f1 = sum(e["f1"] for e in eval_results) / len(eval_results) if eval_results else 0
    print(f"\n  Average F1 Score: {avg_f1:.2f}")

    return {"evaluations": eval_results, "average_f1": round(avg_f1, 3)}


# =============================================================================
# MAIN — Run the full lab workflow
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  LAB 5.1 — Azure Computer Vision API")
    print("  Module 3 | Chitkara University")
    print("=" * 60)

    # ── STEP 1: Connect to Azure ──────────────────────────────────────────────
    client = create_client()
    if client is None:
        print("\n[DEMO MODE] Running demo output since no credentials set.")
        print("            Replace AZURE_ENDPOINT and AZURE_KEY to run live.")
        exit(0)

    # ── STEP 2: Analyze a single image ────────────────────────────────────────
    print("\n[STEP 2] Analyzing single image...")
    single_result = analyze_image_url(
        client,
        SAMPLE_IMAGE_URLS[0],
        confidence_threshold=0.85
    )

    # ── STEP 3: Generate caption ──────────────────────────────────────────────
    print("\n[STEP 3] Caption for first image:")
    if single_result.get("captions"):
        print(f"  → {single_result['captions'][0]['text']}")

    # ── STEP 4: Batch analyze 10 images ───────────────────────────────────────
    # Use the 5 sample URLs twice (or replace with your own 10 image URLs)
    ten_images = (SAMPLE_IMAGE_URLS * 2)[:10]
    print(f"\n[STEP 4] Batch analysis of {len(ten_images)} images...")
    all_results = run_batch_analysis(
        client,
        ten_images,
        confidence_threshold=0.85,
        output_file="lab5_1_results.json"
    )

    # ── STEP 5: Manual accuracy evaluation for 3 images ───────────────────────
    print("\n[STEP 5] Accuracy evaluation...")
    eval_report = evaluate_tag_accuracy(all_results, num_to_evaluate=3)
    with open("lab5_1_evaluation.json", "w") as f:
        json.dump(eval_report, f, indent=2)
    print(f"\n[INFO] Evaluation saved to: lab5_1_evaluation.json")

    # ── STEP 6: Extract text from an image with the Read API ──────────────────
    # text_url = "https://raw.githubusercontent.com/Azure-Samples/cognitive-services-sample-data-files/master/ComputerVision/Images/printed_text.jpg"
    # extracted_text = extract_text_from_image_url(client, text_url)
    # print("\nExtracted Text:\n", extracted_text)

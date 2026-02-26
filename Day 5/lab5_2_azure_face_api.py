# =============================================================================
# LAB 5.2: Azure Face API
# Module 3 | Chitkara University | B.Tech AI Specialization
# Theme: Detect faces, register them in a cloud person group, verify and
#        identify identities using Azure Face API.
# =============================================================================

# ── INSTALL (run once in terminal) ───────────────────────────────────────────
# pip install azure-cognitiveservices-vision-face msrest requests

# ── SETUP: Get your credentials from Azure Portal ────────────────────────────
# 1. Go to https://portal.azure.com
# 2. Create a "Face" resource (Free tier F0)
# 3. Go to resource → "Keys and Endpoint"
# 4. Copy Key 1 and Endpoint into the constants below

from azure.cognitiveservices.vision.face import FaceClient
from azure.cognitiveservices.vision.face.models import (
    TrainingStatusType, Person, QualityForRecognition, DetectionModel,
    RecognitionModel, FaceAttributeType
)
from msrest.authentication import CognitiveServicesCredentials
import json
import time
import os
import requests

# =============================================================================
# CONFIGURATION — Replace with your Azure credentials
# =============================================================================

AZURE_FACE_ENDPOINT = "https://YOUR_RESOURCE_NAME.cognitiveservices.azure.com/"
AZURE_FACE_KEY      = "YOUR_FACE_API_KEY_HERE"

# Person group ID — lowercase, no spaces (unique per Azure resource)
PERSON_GROUP_ID = "chitkara-lab-group"

# Recognition model — recognition_04 is the most accurate (2024)
RECOGNITION_MODEL = RecognitionModel.recognition04
DETECTION_MODEL   = DetectionModel.detection03


# =============================================================================
# SECTION 1: CONNECT TO AZURE FACE API
# =============================================================================

def create_face_client() -> FaceClient:
    """
    Creates and returns an authenticated Azure Face API client.
    """
    if "YOUR_" in AZURE_FACE_ENDPOINT or "YOUR_" in AZURE_FACE_KEY:
        print("[ERROR] Replace AZURE_FACE_ENDPOINT and AZURE_FACE_KEY with your credentials.")
        return None

    client = FaceClient(
        endpoint=AZURE_FACE_ENDPOINT,
        credentials=CognitiveServicesCredentials(AZURE_FACE_KEY)
    )
    print("[INFO] Azure Face API client created successfully.")
    return client


# =============================================================================
# SECTION 2: DETECT FACES — Location + attributes from an image
# =============================================================================

def detect_faces_in_image(face_client: FaceClient, image_url: str) -> list:
    """
    Detects all faces in an image URL and returns their attributes.
    Attributes include: age estimate, emotion, gender, head pose, glasses.
    """
    print(f"\n[DETECT] Image: {image_url[:70]}...")

    try:
        detected_faces = face_client.face.detect_with_url(
            url=image_url,
            detection_model=DETECTION_MODEL,
            recognition_model=RECOGNITION_MODEL,
            return_face_id=True,
            return_face_attributes=[
                FaceAttributeType.age,
                FaceAttributeType.gender,
                FaceAttributeType.emotion,
                FaceAttributeType.glasses,
                FaceAttributeType.head_pose,
                FaceAttributeType.smile,
            ]
        )

        print(f"  Detected {len(detected_faces)} face(s)")

        results = []
        for i, face in enumerate(detected_faces):
            rect = face.face_rectangle
            attrs = face.face_attributes

            # ── Find dominant emotion ──────────────────────────────────────────
            dominant_emotion = "unknown"
            if attrs and attrs.emotion:
                emotion_dict = {
                    "anger": attrs.emotion.anger,
                    "contempt": attrs.emotion.contempt,
                    "disgust": attrs.emotion.disgust,
                    "fear": attrs.emotion.fear,
                    "happiness": attrs.emotion.happiness,
                    "neutral": attrs.emotion.neutral,
                    "sadness": attrs.emotion.sadness,
                    "surprise": attrs.emotion.surprise
                }
                dominant_emotion = max(emotion_dict, key=emotion_dict.get)

            face_data = {
                "face_id": face.face_id,
                "bounding_box": {
                    "top": rect.top,
                    "left": rect.left,
                    "width": rect.width,
                    "height": rect.height
                },
                "age_estimate": round(attrs.age, 1) if attrs and attrs.age else None,
                "gender": attrs.gender.value if attrs and attrs.gender else None,
                "dominant_emotion": dominant_emotion,
                "smile": round(attrs.smile, 3) if attrs and attrs.smile is not None else None,
                "glasses": attrs.glasses.value if attrs and attrs.glasses else None,
            }

            print(f"\n  Face {i+1}:")
            print(f"    Position   : top={rect.top}, left={rect.left}, {rect.width}×{rect.height}px")
            if attrs:
                print(f"    Age est.   : {face_data['age_estimate']}")
                print(f"    Gender     : {face_data['gender']}")
                print(f"    Emotion    : {dominant_emotion}")
                print(f"    Glasses    : {face_data['glasses']}")

            results.append(face_data)

        return results

    except Exception as e:
        print(f"  [ERROR] Detection failed: {e}")
        return []


# =============================================================================
# SECTION 3: SETUP PERSON GROUP — Register people in Azure cloud
# =============================================================================

def create_person_group(face_client: FaceClient, group_id: str, group_name: str = "Lab Group"):
    """
    Creates a new person group in Azure.
    A person group is a container for registered people (like a face database).
    """
    try:
        face_client.person_group.create(
            person_group_id=group_id,
            name=group_name,
            recognition_model=RECOGNITION_MODEL
        )
        print(f"[INFO] Created person group: '{group_name}' (ID: {group_id})")
    except Exception as e:
        if "PersonGroupExists" in str(e):
            print(f"[INFO] Person group '{group_id}' already exists.")
        else:
            print(f"[ERROR] Could not create group: {e}")


def delete_person_group(face_client: FaceClient, group_id: str):
    """Deletes a person group and all its data. Use to reset."""
    try:
        face_client.person_group.delete(person_group_id=group_id)
        print(f"[INFO] Deleted person group: {group_id}")
    except Exception as e:
        print(f"[ERROR] Could not delete group: {e}")


# =============================================================================
# SECTION 4: REGISTER PEOPLE — Add persons and their face photos
# =============================================================================

def register_person(face_client: FaceClient, group_id: str,
                     person_name: str, photo_urls: list) -> str:
    """
    Registers a person in the group with multiple face photos.

    Args:
        face_client   : Authenticated FaceClient
        group_id      : ID of the person group
        person_name   : Display name (e.g., "Alice Smith")
        photo_urls    : List of image URLs showing this person's face (min 2-3 for accuracy)

    Returns:
        person_id (str): Azure-assigned unique ID for this person
    """
    print(f"\n[REGISTER] Adding person: '{person_name}' with {len(photo_urls)} photo(s)")

    try:
        # Create the person record
        person = face_client.person_group_person.create(
            person_group_id=group_id,
            name=person_name
        )
        person_id = person.person_id
        print(f"  Person ID: {person_id}")

        # Add each face photo
        added = 0
        for i, url in enumerate(photo_urls):
            try:
                face_client.person_group_person.add_face_from_url(
                    person_group_id=group_id,
                    person_id=person_id,
                    url=url,
                    detection_model=DETECTION_MODEL
                )
                print(f"  ✓ Added photo {i+1}/{len(photo_urls)}")
                added += 1
                time.sleep(0.2)  # Rate limit buffer

            except Exception as e:
                print(f"  ✗ Photo {i+1} failed: {e}")

        print(f"  Registered '{person_name}' with {added} photos")
        return str(person_id)

    except Exception as e:
        print(f"  [ERROR] Could not register person: {e}")
        return None


def register_sample_people(face_client: FaceClient, group_id: str) -> dict:
    """
    Registers 3 sample people using publicly available face images.
    In a real lab, replace these URLs with actual classmate photos.

    Returns: dict mapping person_name → person_id
    """
    # Sample public face image URLs (Microsoft's official sample images)
    sample_people = {
        "Person_A": [
            "https://raw.githubusercontent.com/Azure-Samples/cognitive-services-sample-data-files/master/Face/images/Family1-Dad1.jpg",
            "https://raw.githubusercontent.com/Azure-Samples/cognitive-services-sample-data-files/master/Face/images/Family1-Dad2.jpg",
        ],
        "Person_B": [
            "https://raw.githubusercontent.com/Azure-Samples/cognitive-services-sample-data-files/master/Face/images/Family1-Mom1.jpg",
            "https://raw.githubusercontent.com/Azure-Samples/cognitive-services-sample-data-files/master/Face/images/Family1-Mom2.jpg",
        ],
        "Person_C": [
            "https://raw.githubusercontent.com/Azure-Samples/cognitive-services-sample-data-files/master/Face/images/Family1-Son1.jpg",
            "https://raw.githubusercontent.com/Azure-Samples/cognitive-services-sample-data-files/master/Face/images/Family1-Son2.jpg",
        ],
    }

    person_id_map = {}
    for name, urls in sample_people.items():
        pid = register_person(face_client, group_id, name, urls)
        if pid:
            person_id_map[name] = pid

    return person_id_map


# =============================================================================
# SECTION 5: TRAIN PERSON GROUP — Must be done before identification
# =============================================================================

def train_person_group(face_client: FaceClient, group_id: str):
    """
    Triggers training of the person group.
    Azure builds face recognition models from the registered photos.
    Must be called after adding/updating any face photos.
    """
    print(f"\n[TRAIN] Training person group: {group_id}")
    face_client.person_group.train(person_group_id=group_id)

    # Poll until training completes
    while True:
        status = face_client.person_group.get_training_status(person_group_id=group_id)
        print(f"  Status: {status.status.value}")
        if status.status == TrainingStatusType.succeeded:
            print("  ✓ Training complete!")
            break
        elif status.status == TrainingStatusType.failed:
            print(f"  ✗ Training failed: {status.message}")
            break
        time.sleep(1)


# =============================================================================
# SECTION 6: IDENTIFY FACES — Who is in this photo?
# =============================================================================

def identify_person_in_image(face_client: FaceClient, group_id: str,
                               image_url: str, person_id_map: dict) -> list:
    """
    Identifies who appears in an image by comparing against the person group.

    Args:
        face_client   : Azure FaceClient
        group_id      : Person group to search in
        image_url     : URL of image to identify
        person_id_map : Dict mapping person_name → person_id (from registration)

    Returns:
        List of identification results with name and confidence
    """
    print(f"\n[IDENTIFY] Image: {image_url[:70]}...")

    # Reverse map: person_id → person_name
    id_to_name = {v: k for k, v in person_id_map.items()}

    try:
        # Step 1: Detect faces
        detected_faces = face_client.face.detect_with_url(
            url=image_url,
            detection_model=DETECTION_MODEL,
            recognition_model=RECOGNITION_MODEL,
            return_face_id=True
        )

        if not detected_faces:
            print("  No faces detected in image.")
            return []

        face_ids = [str(face.face_id) for face in detected_faces]
        print(f"  Detected {len(face_ids)} face(s)")

        # Step 2: Identify against person group
        identify_results = face_client.face.identify(
            face_ids=face_ids,
            person_group_id=group_id,
            max_num_of_candidates_returned=1,  # Top 1 match per face
            confidence_threshold=0.5           # Minimum confidence to consider a match
        )

        # Step 3: Map results back to names
        identifications = []
        for result in identify_results:
            if result.candidates:
                best_candidate = result.candidates[0]
                person_id_str = str(best_candidate.person_id)
                person_name = id_to_name.get(person_id_str, f"Unknown ID: {person_id_str}")
                confidence = round(best_candidate.confidence, 3)
                status = "✓ MATCH" if confidence >= 0.7 else "? LOW CONF"
                print(f"  {status} → '{person_name}' (confidence: {confidence})")
                identifications.append({
                    "face_id": str(result.face_id),
                    "identified_as": person_name,
                    "confidence": confidence,
                    "is_match": confidence >= 0.7
                })
            else:
                print("  → Unknown person (no match found)")
                identifications.append({
                    "face_id": str(result.face_id),
                    "identified_as": "Unknown",
                    "confidence": 0.0,
                    "is_match": False
                })

        return identifications

    except Exception as e:
        print(f"  [ERROR] Identification failed: {e}")
        return []


# =============================================================================
# SECTION 7: VERIFY — Are these two face IDs the same person?
# =============================================================================

def verify_faces(face_client: FaceClient,
                  image_url_1: str, image_url_2: str) -> dict:
    """
    Verifies whether two photos show the same person.
    Returns verification result with confidence score.
    """
    print(f"\n[VERIFY] Comparing two images...")

    try:
        # Detect face in each image
        face1 = face_client.face.detect_with_url(
            url=image_url_1,
            detection_model=DETECTION_MODEL,
            recognition_model=RECOGNITION_MODEL
        )
        face2 = face_client.face.detect_with_url(
            url=image_url_2,
            detection_model=DETECTION_MODEL,
            recognition_model=RECOGNITION_MODEL
        )

        if not face1 or not face2:
            print("  [ERROR] Could not detect face in one or both images.")
            return {}

        result = face_client.face.verify_face_to_face(
            face_id1=face1[0].face_id,
            face_id2=face2[0].face_id
        )

        same = result.is_identical
        confidence = round(result.confidence, 3)
        print(f"  Same person? {'✓ YES' if same else '✗ NO'}")
        print(f"  Confidence : {confidence}")
        return {"is_same_person": same, "confidence": confidence}

    except Exception as e:
        print(f"  [ERROR] Verification failed: {e}")
        return {}


# =============================================================================
# SECTION 8: ACCURACY EVALUATION
# =============================================================================

def evaluate_identification_accuracy(face_client, group_id, person_id_map):
    """
    Tests identification accuracy on test images.
    Uses Microsoft's official sample test images.
    """
    # Test images: (url, expected_person_name)
    test_cases = [
        ("https://raw.githubusercontent.com/Azure-Samples/cognitive-services-sample-data-files/master/Face/images/Family1-Dad3.jpg", "Person_A"),
        ("https://raw.githubusercontent.com/Azure-Samples/cognitive-services-sample-data-files/master/Face/images/Family1-Mom3.jpg", "Person_B"),
        ("https://raw.githubusercontent.com/Azure-Samples/cognitive-services-sample-data-files/master/Face/images/Family1-Son2.jpg", "Person_C"),
    ]

    print(f"\n[EVAL] Evaluating accuracy on {len(test_cases)} test images")
    print("-" * 60)

    correct = 0
    total = len(test_cases)
    eval_log = []

    for img_url, expected_name in test_cases:
        results = identify_person_in_image(face_client, group_id, img_url, person_id_map)
        if results:
            predicted = results[0]["identified_as"]
            is_correct = predicted == expected_name
            correct += int(is_correct)
            status = "✓" if is_correct else "✗"
            print(f"  {status} Expected: {expected_name:10s} | Predicted: {predicted}")
            eval_log.append({
                "image": img_url.split("/")[-1],
                "expected": expected_name,
                "predicted": predicted,
                "confidence": results[0]["confidence"],
                "correct": is_correct
            })
        else:
            print(f"  ✗ Expected: {expected_name} | No face detected")
            eval_log.append({
                "image": img_url.split("/")[-1],
                "expected": expected_name,
                "predicted": "No face",
                "confidence": 0,
                "correct": False
            })

    accuracy = correct / total
    print("-" * 60)
    print(f"\n[RESULT] Accuracy: {correct}/{total} = {accuracy*100:.1f}%")

    report = {
        "correct": correct,
        "total": total,
        "accuracy_percent": round(accuracy * 100, 1),
        "details": eval_log
    }
    with open("lab5_2_accuracy_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("[INFO] Report saved to: lab5_2_accuracy_report.json")

    return accuracy


# =============================================================================
# MAIN — Run the full lab workflow
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  LAB 5.2 — Azure Face API")
    print("  Module 3 | Chitkara University")
    print("=" * 60)

    # ── STEP 1: Connect ───────────────────────────────────────────────────────
    face_client = create_face_client()
    if face_client is None:
        print("\n[DEMO MODE] Replace credentials to run live.")
        exit(0)

    # ── STEP 2: Detect faces in a group photo ─────────────────────────────────
    print("\n[STEP 2] Face detection with attributes...")
    group_photo_url = "https://raw.githubusercontent.com/Azure-Samples/cognitive-services-sample-data-files/master/Face/images/identification1.jpg"
    detected = detect_faces_in_image(face_client, group_photo_url)

    # ── STEP 3: Create person group ───────────────────────────────────────────
    print("\n[STEP 3] Setting up person group...")
    # Uncomment the next line to reset/delete existing group first:
    # delete_person_group(face_client, PERSON_GROUP_ID)
    create_person_group(face_client, PERSON_GROUP_ID, "Chitkara Lab Demo Group")

    # ── STEP 4: Register 3 people ─────────────────────────────────────────────
    print("\n[STEP 4] Registering people...")
    person_id_map = register_sample_people(face_client, PERSON_GROUP_ID)
    print(f"\n  Registered people: {list(person_id_map.keys())}")

    # Save the map so you don't lose it between runs
    with open("person_id_map.json", "w") as f:
        json.dump(person_id_map, f, indent=2)
    print("  Saved person_id_map.json")

    # ── STEP 5: Train the group ───────────────────────────────────────────────
    print("\n[STEP 5] Training person group...")
    train_person_group(face_client, PERSON_GROUP_ID)

    # ── STEP 6: Identify faces in a test photo ────────────────────────────────
    print("\n[STEP 6] Identifying faces...")
    test_url = "https://raw.githubusercontent.com/Azure-Samples/cognitive-services-sample-data-files/master/Face/images/identification1.jpg"
    identify_results = identify_person_in_image(face_client, PERSON_GROUP_ID,
                                                 test_url, person_id_map)

    # ── STEP 7: Verify two faces ──────────────────────────────────────────────
    print("\n[STEP 7] Face verification...")
    url1 = "https://raw.githubusercontent.com/Azure-Samples/cognitive-services-sample-data-files/master/Face/images/Family1-Dad1.jpg"
    url2 = "https://raw.githubusercontent.com/Azure-Samples/cognitive-services-sample-data-files/master/Face/images/Family1-Dad3.jpg"
    verify_faces(face_client, url1, url2)

    # ── STEP 8: Evaluate accuracy ─────────────────────────────────────────────
    print("\n[STEP 8] Accuracy evaluation...")
    evaluate_identification_accuracy(face_client, PERSON_GROUP_ID, person_id_map)

    print("\n[DONE] Lab 5.2 complete!")

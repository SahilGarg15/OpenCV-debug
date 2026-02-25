# =============================================================================
# LAB 4.1: Face Detection & Recognition with DeepFace
# Module 3 | Chitkara University | B.Tech AI Specialization
# Theme: Build a face recognition system that registers 5 people,
#        detects faces from webcam, and identifies them.
# Target: ≥ 90% recognition accuracy on test photos
# =============================================================================

# ── INSTALL (run once in terminal) ───────────────────────────────────────────
# pip install deepface opencv-python

import os
import cv2
import json
import time
import numpy as np
from deepface import DeepFace

# =============================================================================
# SECTION 1: SETUP — Create the database folder structure
# =============================================================================
# Expected structure:
#   database/
#     alice/   → alice1.jpg, alice2.jpg, alice3.jpg, alice4.jpg, alice5.jpg
#     bob/     → bob1.jpg, ...
#     carol/   → ...
#     david/   → ...
#     eve/     → ...

DATABASE_PATH = "./database"
os.makedirs(DATABASE_PATH, exist_ok=True)
print(f"[INFO] Database folder ready at: {os.path.abspath(DATABASE_PATH)}")
print("[INFO] Add subfolders with photos: database/person_name/image1.jpg")


# =============================================================================
# SECTION 2: CAPTURE PHOTOS FOR DATABASE (Optional helper)
# Uses webcam to capture N photos per person and save to database folder.
# Skip this section if you already have photos ready.
# =============================================================================

def capture_face_photos(person_name: str, num_photos: int = 5):
    """
    Opens webcam and captures `num_photos` images for a given person.
    Saves them to database/<person_name>/
    Press SPACE to capture each photo, ESC to exit early.
    """
    save_dir = os.path.join(DATABASE_PATH, person_name)
    os.makedirs(save_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam. Check connection.")
        return

    print(f"\n[INFO] Capturing {num_photos} photos for '{person_name}'")
    print("       Press SPACE to capture | ESC to stop early")

    count = 0
    while count < num_photos:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()
        cv2.putText(display,
                    f"Person: {person_name} | Photo {count+1}/{num_photos}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(display, "SPACE = capture | ESC = quit",
                    (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        cv2.imshow("Capture Face", display)

        key = cv2.waitKey(1)
        if key == 27:   # ESC
            break
        elif key == 32: # SPACE
            filename = os.path.join(save_dir, f"{person_name}{count+1}.jpg")
            cv2.imwrite(filename, frame)
            print(f"  ✓ Saved: {filename}")
            count += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Captured {count} photos for '{person_name}'")


# =============================================================================
# SECTION 3: FACE VERIFICATION — Are these two photos the same person?
# =============================================================================

def verify_two_faces(img_path1: str, img_path2: str):
    """
    Uses DeepFace.verify() to check if two images show the same person.
    Returns: dict with 'verified' (bool) and 'distance' (float)
    """
    print(f"\n[VERIFY] Comparing:\n  Image 1: {img_path1}\n  Image 2: {img_path2}")
    try:
        result = DeepFace.verify(
            img1_path=img_path1,
            img2_path=img_path2,
            model_name="VGG-Face",      # Backbone: VGG-Face (robust, default)
            distance_metric="cosine",   # cosine distance for comparison
            enforce_detection=True      # Raise error if no face found
        )
        print(f"  Same person? {'✓ YES' if result['verified'] else '✗ NO'}")
        print(f"  Distance   : {result['distance']:.4f}  (threshold: {result['threshold']:.4f})")
        print(f"  Model      : {result['model']}")
        return result
    except Exception as e:
        print(f"  [ERROR] Verification failed: {e}")
        return None


# =============================================================================
# SECTION 4: FACE IDENTIFICATION — Who is this person?
# Compares a query image against everyone in the database folder.
# =============================================================================

def identify_face(query_image_path: str):
    """
    Uses DeepFace.find() to search the registered database for the closest match.
    Returns the best matching person name and distance score.
    """
    print(f"\n[IDENTIFY] Searching database for: {query_image_path}")
    try:
        results_df = DeepFace.find(
            img_path=query_image_path,
            db_path=DATABASE_PATH,
            model_name="VGG-Face",
            distance_metric="cosine",
            enforce_detection=True,
            silent=True             # Suppress progress bars
        )

        # results_df is a list of DataFrames (one per face detected)
        if results_df and len(results_df[0]) > 0:
            top_match = results_df[0].iloc[0]
            # Extract person name from file path
            matched_path = top_match["identity"]
            person_name = os.path.basename(os.path.dirname(matched_path))
            distance = top_match["distance"]
            print(f"  Best match : {person_name}")
            print(f"  Distance   : {distance:.4f}")
            return person_name, distance
        else:
            print("  No match found in database.")
            return "Unknown", 1.0

    except Exception as e:
        print(f"  [ERROR] Identification failed: {e}")
        return "Unknown", 1.0


# =============================================================================
# SECTION 5: REAL-TIME WEBCAM RECOGNITION
# Captures frames, detects face, searches database, displays name.
# =============================================================================

def run_realtime_recognition(confidence_threshold: float = 0.4):
    """
    Real-time face recognition loop using webcam.
    Compares each captured frame against the registered database.
    Press 'q' to quit | 'c' to capture and identify current frame.
    """
    if not os.path.exists(DATABASE_PATH) or not os.listdir(DATABASE_PATH):
        print("[ERROR] Database is empty. Add face photos first.")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return

    print("\n[INFO] Real-time recognition started.")
    print("       Press 'c' to identify current face | 'q' to quit")

    last_result = ("Press 'c' to identify", None)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()

        # ── Detect face bounding box for display ──
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1,
                                               minNeighbors=5, minSize=(80, 80))
        for (x, y, w, h) in faces:
            cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # ── Display last identification result ──
        name, dist = last_result
        label_color = (0, 255, 0) if name not in ["Unknown", "Press 'c' to identify"] else (0, 0, 255)
        cv2.putText(display, f"ID: {name}",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, label_color, 2)
        if dist is not None:
            cv2.putText(display, f"Distance: {dist:.3f}",
                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

        cv2.putText(display, "c=capture  q=quit",
                    (10, display.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        cv2.imshow("Face Recognition — Module 3 Lab 4.1", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            # Save frame temporarily and identify
            temp_path = "/tmp/temp_query.jpg"
            cv2.imwrite(temp_path, frame)
            print("\n[INFO] Identifying face...")
            name, dist = identify_face(temp_path)
            last_result = (name, dist)

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Recognition session ended.")


# =============================================================================
# SECTION 6: ACCURACY EVALUATION
# Tests recognition on a set of labelled test images and reports accuracy.
# =============================================================================

def evaluate_accuracy(test_folder: str):
    """
    Evaluates recognition accuracy on a test set.

    Expected test_folder structure:
      test_images/
        alice/  → alice_test1.jpg, alice_test2.jpg ...
        bob/    → bob_test1.jpg, ...
        ...

    Returns: accuracy score (0.0 to 1.0)
    """
    if not os.path.exists(test_folder):
        print(f"[ERROR] Test folder not found: {test_folder}")
        return

    correct = 0
    total = 0
    results_log = []

    print(f"\n[EVAL] Evaluating accuracy on test set: {test_folder}")
    print("-" * 60)

    for person_name in os.listdir(test_folder):
        person_dir = os.path.join(test_folder, person_name)
        if not os.path.isdir(person_dir):
            continue

        for img_file in os.listdir(person_dir):
            if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            img_path = os.path.join(person_dir, img_file)
            predicted_name, distance = identify_face(img_path)

            match = predicted_name.lower() == person_name.lower()
            correct += int(match)
            total += 1

            status = "✓" if match else "✗"
            print(f"  {status}  True: {person_name:15s} | Predicted: {predicted_name:15s} | Dist: {distance:.3f}")
            results_log.append({
                "image": img_file,
                "true_label": person_name,
                "predicted": predicted_name,
                "distance": round(float(distance), 4),
                "correct": match
            })

    accuracy = correct / total if total > 0 else 0
    print("-" * 60)
    print(f"\n[RESULT] Accuracy: {correct}/{total} = {accuracy*100:.1f}%")
    print(f"         Target  : ≥ 90%  |  {'✓ PASSED' if accuracy >= 0.9 else '✗ Not yet — try more training photos'}")

    # Save results to JSON
    report = {
        "total_images": total,
        "correct": correct,
        "accuracy_percent": round(accuracy * 100, 2),
        "target_percent": 90,
        "passed": accuracy >= 0.9,
        "details": results_log
    }
    with open("lab4_1_accuracy_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\n[INFO] Full report saved to: lab4_1_accuracy_report.json")
    return accuracy


# =============================================================================
# MAIN — Run the full lab workflow
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  LAB 4.1 — Face Detection & Recognition")
    print("  Module 3 | Chitkara University")
    print("=" * 60)

    # ── STEP 1: Capture photos (comment out if you already have photos) ──────
    # Uncomment and run once per person to build your database:
    # for name in ["alice", "bob", "carol", "david", "eve"]:
    #     capture_face_photos(name, num_photos=5)

    # ── STEP 2: Test face verification (same vs different person) ────────────
    # Replace with actual paths to your images:
    # verify_two_faces("database/alice/alice1.jpg", "database/alice/alice2.jpg")
    # verify_two_faces("database/alice/alice1.jpg", "database/bob/bob1.jpg")

    # ── STEP 3: Identify a single image ──────────────────────────────────────
    # identify_face("database/alice/alice1.jpg")

    # ── STEP 4: Run real-time webcam recognition ──────────────────────────────
    print("\n[STEP] Starting real-time recognition...")
    print("       Ensure database/ folder has at least 1 registered person.\n")
    # run_realtime_recognition()

    # ── STEP 5: Evaluate accuracy on test images ──────────────────────────────
    # evaluate_accuracy("./test_images")

    print("\n[INFO] Uncomment the steps above to run them in sequence.")
    print("       Recommended order: capture → verify → identify → live → evaluate")

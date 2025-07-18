import time
import os
import firebase_admin
from firebase_admin import credentials, storage, firestore
from google.cloud.storage.retry import DEFAULT_RETRY
import board
import neopixel_spi as neopixel

# Configuration Constants
NUM_LEDS = 32
PIXEL_ORDER = neopixel.GRBW
LIGHT_ON_COLOR = (255, 255, 255, 255)
LIGHT_OFF_COLOR = (0, 0, 0, 0)
FIREBASE_STORAGE_FOLDER = 'testing/'
SERVICE_ACCOUNT_KEY_PATH = 'score-trap-mqtt-test-firebase-adminsdk-wpt0n-1d8ecd74df.json'

# Define separate focus parameters for each camera
# FOCUS_PARAMS = {
    # 1: {"starting_focus": 11.26, "focus_jump": 0.01, "total_images": 10},
    # 0: {"starting_focus": 11.1, "focus_jump": 0.01, "total_images": 10}
# }
FOCUS_PARAMS = {
    1: {"starting_focus": 11.26, "focus_jump": 0.01, "total_images": 10},
    0: {"starting_focus": 10.95, "focus_jump": 0.01, "total_images": 10}
}

# Initialize Firebase App and Firestore
cred = credentials.Certificate(SERVICE_ACCOUNT_KEY_PATH)
firebase_admin.initialize_app(cred, {
    'storageBucket': 'score-trap-mqtt-test.appspot.com'
})
db = firestore.client()

# Initialize NeoPixel (only once)
pixels = neopixel.NeoPixel_SPI(board.SPI(),
                               NUM_LEDS,
                               pixel_order=PIXEL_ORDER,
                               auto_write=True)


def changeLightColor(color):
    """Change the color of the NeoPixel LEDs."""
    for i in range(NUM_LEDS):
        pixels[i] = color
    print(f"LEDs set to: {color}")

def upload_image(image_path, destination_blob_name, retries=5, delay=5):
    """
    Uploads an image to Firebase Storage.
    Returns a metadata dictionary on success.
    """
    bucket = storage.bucket()
    blob = bucket.blob(destination_blob_name)

    for attempt in range(retries):
        try:
            if not os.path.exists(image_path):
                print(f"File {image_path} does not exist.")
                return None

            blob.upload_from_filename(image_path, timeout=300, retry=DEFAULT_RETRY)
            blob.make_public()  # Optionally make the image public
            download_url = blob.public_url

            print(f"Uploaded {image_path} to {destination_blob_name}")
            print(f"Public URL: {download_url}")

            # Return metadata for later aggregation
            return {
                "fileName": os.path.basename(image_path),
                "filePath": destination_blob_name,
                "downloadURL": download_url
            }
        except Exception as e:
            print(f"Attempt {attempt + 1} failed for {image_path}: {e}")
            if attempt < retries - 1:
                time.sleep(delay)
                delay *= 2  # Exponential backoff
            else:
                print("All attempts failed.")
                raise

def capture_image(focus, file_name, camera_number):
    """
    Captures a single image using libcamera-still with a given focus.
    """
    cmd = f"libcamera-still --nopreview -t 1 --lens-position {focus} --camera {camera_number} -o {file_name}"
    print(f"Capturing image: {cmd}")
    os.system(cmd)

def capture_images(file_name_prefix, focus_levels, camera_number):
    """
    Captures images at the specified focus levels for a given camera.
    Returns a list of local file paths.
    """
    file_names = []
    for i, focus in enumerate(focus_levels):
        file_name = f"{file_name_prefix}_focus_{i}.jpg"
        capture_image(focus, file_name, camera_number)
        file_names.append(file_name)
    return file_names

def main():
    time_start = time.time()
    print("Capture and upload process started.")

    # Turn on the LED (or light) before capture
    changeLightColor(LIGHT_ON_COLOR)

    # Get a unique event identifier from the user (e.g., door number or event id)
    unique_identifier = input("Enter the unique identifier: ").strip()

    # Create a local directory to temporarily store captured images
    local_dir = os.path.join(os.getcwd(), unique_identifier)
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)

    # Build a Firebase Storage subfolder path using the unique identifier
    firebase_subfolder = f"{FIREBASE_STORAGE_FOLDER}{unique_identifier}/"

    # List to accumulate metadata for all images in this event
    aggregated_metadata = []

    # Process images from both cameras (e.g., camera 1 and camera 0)
    for camera_number in [1, 0]:
        print(f"\nProcessing camera {camera_number}...")
        file_name_prefix = os.path.join(local_dir, f"image_{unique_identifier}_camera_{camera_number}")

        # Retrieve focus parameters for this camera
        params = FOCUS_PARAMS.get(camera_number)
        if not params:
            print(f"No focus parameters defined for camera {camera_number}. Skipping...")
            continue

        # Create the focus levels list for this camera
        focus_levels = [
            params["starting_focus"] + i * params["focus_jump"]
            for i in range(params["total_images"])
        ]
        print(f"Focus levels for camera {camera_number}: {focus_levels}")

        # Capture the images at the specified focus levels
        captured_files = capture_images(file_name_prefix, focus_levels, camera_number)

        # Upload each captured image and collect its metadata
        for file_path in captured_files:
            destination_blob_name = f"{firebase_subfolder}{os.path.basename(file_path)}"
            metadata = upload_image(file_path, destination_blob_name)
            if metadata is not None:
                # Optionally add camera information to the metadata
                metadata["camera"] = camera_number
                aggregated_metadata.append(metadata)

    # Turn off the LED/light after capture
    changeLightColor(LIGHT_OFF_COLOR)

    # Aggregate all image metadata into one document and upload to Firestore.
    event_metadata = {
        "uniqueIdentifier": unique_identifier,
        "images": aggregated_metadata,
        "timestamp": firestore.SERVER_TIMESTAMP
    }
    
    try:
        # Create a new document with an auto-generated ID
        doc_ref = db.collection("uploads").document()
        doc_ref.set(event_metadata)
        print(f"Aggregated metadata uploaded for id '{unique_identifier}' with {len(aggregated_metadata)} images.")
        print(f"Firestore document ID: {doc_ref.id}")

        # Verify by fetching the document immediately after writing
        doc_snapshot = doc_ref.get()
        if doc_snapshot.exists:
            print("Verification: Document successfully written!")
        else:
            print("Verification: Document not found after writing.")
    except Exception as e:
        print("Error writing document to Firestore:", e)

    total_time = time.time() - time_start
    print(f"\nProcess completed in {total_time:.2f} seconds.")

if __name__ == "__main__":
    main()

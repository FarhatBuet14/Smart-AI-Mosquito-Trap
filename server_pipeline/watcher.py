import firebase_admin
from firebase_admin import credentials, firestore, storage
import requests
import os
import time
import cv2
import numpy as np
import gc

# Initialize Firebase Admin SDK
SERVICE_ACCOUNT_KEY_PATH = 'score-trap-mqtt-test-firebase-adminsdk-wpt0n-1d8ecd74df.json'
if not firebase_admin._apps:
    cred = credentials.Certificate(SERVICE_ACCOUNT_KEY_PATH)
    firebase_admin.initialize_app(cred, {
        'storageBucket': 'score-trap-mqtt-test.appspot.com'
    })

# Reference to Firestore
db = firestore.client()

# Temporary directory for downloaded and processed images
TEMP_IMAGE_DIR = "./temp_image_dir/"
os.makedirs(TEMP_IMAGE_DIR, exist_ok=True)

# Global flag to indicate if the initial snapshot has been processed
initial_snapshot_processed = False

#########################
# Image Processing Code #
#########################

def stack_images(image_paths):
    """
    Loads images from the provided file paths and stacks them by taking the maximum
    intensity value for each pixel.
    """
    images = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is not None:
            images.append(img)
        else:
            print(f"Failed to load image: {path}")
    if len(images) == 0:
        return None
    
    # Stack images using element-wise maximum
    stacked = np.max(np.array(images), axis=0)
    
    # #Simple Median Stack
    # stacked = np.median(np.array(images), axis=0).astype(np.uint8)
    
    # # Focus Measure-Based Per-Pixel Selection (Laplacian Energy)
    # images = [cv2.resize(img, (images[0].shape[1], images[0].shape[0])) for img in images]
    # gray_stack = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images]
    # laplacian_stack = [cv2.Laplacian(gray, cv2.CV_64F)**2 for gray in gray_stack]  # Edge energy

    # focus_map = np.argmax(np.stack(laplacian_stack, axis=0), axis=0)  # Index of sharpest focus
    # h, w = focus_map.shape
    # stacked = np.zeros_like(images[0])

    # for i in range(h):
    #     for j in range(w):
    #         stacked[i, j] = images[focus_map[i, j]][i, j]
    
    return stacked




def sharpen_image(image):
    """
    Sharpens the image using a simple convolution kernel.
    """
    # ## Ahsan sharpening algorithm
    # kernel = np.array([[0, -1, 0],
    #                    [-1, 5, -1],
    #                    [0, -1, 0]])
    
    # # More aggressive sharpening
    # kernel = np.array([[ -1, -1, -1],
    #         [ -1,  9, -1],
    #         [ -1, -1, -1]])
    
    # ## Unsharp Masking kernel (approximate)
    # kernel = np.array([[-1, -2, -1],
    #                    [-2, 28, -2],
    #                    [-1, -2, -1]], dtype=np.float32) / 16
    
    # return cv2.filter2D(image, -1, kernel)
    
    
    # # CLAHE for contrast normalization
    # lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    # l, a, b = cv2.split(lab)
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # cl = clahe.apply(l)
    # limg = cv2.merge((cl, a, b))
    # contrast_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # # Laplacian sharpening
    # laplacian = cv2.Laplacian(contrast_img, cv2.CV_64F)
    # sharpened = cv2.convertScaleAbs(contrast_img - 0.7 * laplacian)
    
    ## Unsharp Masking
    kernel_size=(5,5)
    sigma=1.0
    alpha=2.0
    
    # Gaussian blur
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    # Unsharp mask
    sharpened = cv2.addWeighted(image, 1 + alpha, blurred, -alpha, 0)
    
    
    return sharpened
    
    

################################
# Firebase Storage Upload Code #
################################

def upload_image_listener(local_path, destination_blob_name, retries=5, delay=5):
    """
    Uploads a local file to Firebase Storage and makes it public.
    Returns a metadata dictionary on success.
    """
    bucket = storage.bucket()
    blob = bucket.blob(destination_blob_name)
    
    for attempt in range(retries):
        try:
            if not os.path.exists(local_path):
                print(f"File {local_path} does not exist.")
                return None

            blob.upload_from_filename(local_path, timeout=300)
            blob.make_public()  # Makes the file accessible via a public URL
            download_url = blob.public_url
            print(f"Uploaded processed image '{local_path}' to '{destination_blob_name}'.")
            return {
                "fileName": os.path.basename(local_path),
                "filePath": destination_blob_name,
                "downloadURL": download_url
            }
        except Exception as e:
            print(f"Attempt {attempt + 1} failed for uploading {local_path}: {e}")
            if attempt < retries - 1:
                time.sleep(delay)
                delay *= 2
            else:
                print("All attempts failed for uploading processed image.")
                return None

#############################
# Download Utility Function #
#############################

def download_image(url, file_path):
    """Downloads an image from the given URL and saves it locally."""
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        with open(file_path, 'wb') as file:
            file.write(response.content)
        print(f"Image downloaded and saved to: {file_path}")
    except requests.RequestException as e:
        print(f"Error downloading the image from {url}: {e}")

#########################################
# Local Image Deletion Utility Function #
#########################################

def delete_image(file_path):
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Deleted file: {file_path}")
            return True
        else:
            print(f"File not found: {file_path}")
            return False
    except Exception as e:
        print(f"Error deleting file {file_path}: {e}")
        return False

##############################################
# Process Aggregated Document from Firestore #
##############################################

def process_event_document(doc):
    """
    For the given aggregated document:
      - Groups images by their 'camera' field.
      - Downloads all images.
      - For each camera group, stacks and sharpens the images.
      - Uploads the processed image to Firebase Storage.
      - Updates the document with the processed images' metadata.
      - Deletes the individual downloaded images after processing.
      - Prints timing information for download and processing.
    """
    doc_id = doc.id
    data = doc.to_dict()
    if 'images' not in data:
        print("No aggregated images found in the document; skipping processing.")
        return
    
    unique_identifier = data.get("uniqueIdentifier")
    if not unique_identifier:
        print("Unique identifier not found in the document; defaulting to doc id.")
        unique_identifier = doc_id

    images_metadata = data['images']
    # Group images by camera (e.g., camera1, camera2)
    camera_groups = {}
    for img_meta in images_metadata:
        cam = img_meta.get('camera')
        if cam is None:
            continue
        camera_groups.setdefault(cam, []).append(img_meta)

    processed_images = {}
    for cam, img_list in camera_groups.items():
        print(f"Processing images for camera {cam} ({len(img_list)} images)...")
        local_paths = []
        
        # Time the downloading phase
        download_start = time.time()
        # Download all images for this camera
        for img_meta in img_list:
            file_name = img_meta.get('fileName')
            download_url = img_meta.get('downloadURL')
            if file_name and download_url:
                local_path = os.path.join(TEMP_IMAGE_DIR, file_name)
                download_image(download_url, local_path)
                local_paths.append(local_path)
            else:
                print("Missing fileName or downloadURL in metadata; skipping an image.")
        download_end = time.time()
        print(f"Time taken to download images for camera {cam}: {download_end - download_start:.2f} seconds")
        
        # Time the stacking phase
        stack_start = time.time()
        stacked = stack_images(local_paths)
        stack_end = time.time()
        print(f"Time taken to stack images for camera {cam}: {stack_end - stack_start:.2f} seconds")
        
        if stacked is None:
            print(f"Could not load any images for camera {cam}.")
            continue
        
        # Time the sharpening phase
        sharpen_start = time.time()
        sharpened = sharpen_image(stacked)
        sharpen_end = time.time()
        print(f"Time taken to sharpen image for camera {cam}: {sharpen_end - sharpen_start:.2f} seconds")
        
        # Save the processed image locally
        processed_filename = f"stacked_sharp_camera{cam}_{doc_id}.jpg"
        processed_local_path = os.path.join(TEMP_IMAGE_DIR, processed_filename)
        cv2.imwrite(processed_local_path, sharpened)
        print(f"Processed image saved locally: {processed_local_path}")
        
        # Run process_image.py on the downloaded image, passing the doc_id and filename
        #print(f"Running process_image.py on {processed_filename} with doc_id {doc_id}...")
        os.system(f"python localize_image.py {doc_id} {processed_filename}")
        
        # Upload the processed image to Firebase Storage
        processed_storage_path = f"processed images/{unique_identifier}/{processed_filename}"
        processed_metadata = upload_image_listener(processed_local_path, processed_storage_path)
        if processed_metadata:
            processed_images[f"camera{cam}"] = processed_metadata
            delete_image(processed_local_path)
        
        # Delete the individual downloaded images (keeping the processed one)
        for file_path in local_paths:
            delete_image(file_path)

    # Update the original Firestore document with the processed images metadata.
    db.collection("uploads").document(doc_id).update({
        "processedImages": processed_images,
        "processedTimestamp": firestore.SERVER_TIMESTAMP
    })
    print(f"Updated document {doc_id} with processed images metadata.")

###########################
# Firestore Listener Code #
###########################

def on_snapshot(doc_snapshot, changes, read_time):
    """
    Callback for Firestore snapshots. For newly added documents that contain an
    aggregated 'images' field, process the event document.
    """
    global initial_snapshot_processed

    if not initial_snapshot_processed:
        print("Initial snapshot detected. Marking as processed and ignoring existing documents.")
        initial_snapshot_processed = True
        return

    for change in changes:
        if change.type.name == 'ADDED':
            doc = change.document
            data = doc.to_dict()
            # Look for aggregated metadata (an 'images' array)
            if 'images' in data:
                print(f"New aggregated metadata document detected: {data}")
                process_event_document(doc)
            else:
                print("Document does not contain aggregated metadata; skipping.")

def start_listener():
    print("Starting Firestore listener...")
    uploads_ref = db.collection("uploads")
    query_watch = uploads_ref.on_snapshot(on_snapshot)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Listener stopped.")

#########################
# Main Execution Block  #
#########################

if __name__ == "__main__":
    start_listener()

import os
import sys
import cv2
import numpy as np
from datetime import datetime, timezone
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import firebase_admin
from firebase_admin import credentials, firestore, storage

# ----------- Limiting GPU memory ----------------
tfConfig = tf.compat.v1.ConfigProto()
tfConfig.gpu_options.allow_growth = True
tfConfig.gpu_options.per_process_gpu_memory_fraction = 0.1  # allocating highest 10% GPU memory
sess = tf.compat.v1.Session(config=tfConfig)
tf.compat.v1.keras.backend.set_session(sess)

predicted_name = ""
predicted_probability = 0
mosquitoes = [
    'NotSTep',
    'NotSTep',
    'NotSTep',
    'NotSTep',
    'NotSTep',
    'stephensi',
    'NotSTep',
    'NotSTep',
    'NotSTep'
]
temp_file_path = "./temp_image_dir/"

# Initialize Firebase Admin SDK and Firestore
SERVICE_ACCOUNT_KEY_PATH = 'score-trap-mqtt-test-firebase-adminsdk-wpt0n-1d8ecd74df.json'

if not firebase_admin._apps:
    cred = credentials.Certificate(SERVICE_ACCOUNT_KEY_PATH)
    firebase_admin.initialize_app(cred, {
        'storageBucket': 'score-trap-mqtt-test.appspot.com'
    })

firestore_db = firestore.client()
bucket = storage.bucket()

def get_output_layer(model, layer_name):
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    layer = layer_dict[layer_name]
    return layer

def execute(image_filename):
    global predicted_name, predicted_probability
    model = load_model('./models/adult_species_classifier/0.9977weights.0.8996.hdf5')
    img = load_img(temp_file_path + image_filename, target_size=(299, 299))

    """
    Species prediction
    """
    x = img_to_array(img) / 255
    res = model.predict(np.array([x]).astype('float32')).tolist()[0]
    predicted = mosquitoes[res.index(max(res))]
    percentage = str(round(max(res) * 100, 2))

    print(f"{image_filename} is {predicted} with a probability of {percentage}%")

    """
    CAM generation
    """
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    layer_model = layer_dict['model_1']

    img_original = cv2.imread(temp_file_path + image_filename)
    print("img: " + temp_file_path + image_filename)
    height, width, _ = img_original.shape

    import keras.backend as K
    class_weights = np.array(model.layers[1].layers[-3].get_weights()[0]).reshape(-1, 1)
    final_conv_layer = get_output_layer(model.layers[1], "conv_7b")

    get_output = K.function([model.layers[1].inputs], [final_conv_layer.output])
    conv_outputs = np.array(get_output(np.array([x]).astype('float32')))[0][0]

    cam = np.zeros(dtype=np.float32, shape=conv_outputs.shape[0:2])
    for i, w in enumerate(class_weights[:, 0]):
        cam += w * conv_outputs[:, :, i]
    cam /= np.max(cam)

    cam = cv2.resize(cam, (width, height))
    heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
    heatmap[np.where(cam < 0.2)] = 0
    img_cam = heatmap * 0.5 + img_original

    # cam_filename with predicted class and probability
    base_name = os.path.splitext(image_filename)[0]
    cam_filename = f"{base_name}_cam_{predicted}_{percentage}.jpg"

    print(f"Writing to file: {temp_file_path}{cam_filename}")
    try:
        cv2.imwrite(temp_file_path + cam_filename, img_cam)
    except Exception as e:
        print(f"Error writing to file: {temp_file_path}{cam_filename}, error: {e}")
    predicted_name = predicted
    predicted_probability = percentage

    return cam_filename, predicted_name, predicted_probability

def process_image_prediction(doc_id, image_filename):
    print("\n\nNew image received: ", image_filename)

    # Run prediction and CAM generation
    print("Running prediction ....")
    cam_filename, predicted_class, predicted_probability = execute(image_filename)
    
    ## Future work
    # mos_filename_list = execute(image_filename)
    # mos_images = []
    # for i in range(5):
    #     mos_images.append(cam_filename)

    # Upload CAM image to Firebase Storage using the Admin SDK
    cam_image_url = upload_image_to_storage(cam_filename, "mosquito_localization/")

    print(f"Uploaded CAM image to Firebase Storage: {cam_image_url}")

    # Update the existing Firestore document with processed data
    update_firestore_record(doc_id, predicted_class, predicted_probability, cam_image_url)

    # Removing temporary images (both original and CAM image)
    cleanup_images([image_filename, cam_filename])

    print("Image processing completed.")

def upload_image_to_storage(image_filename, folder_path):
    """Uploads the image file to Firebase Storage and returns its public URL."""
    blob_path = folder_path + image_filename
    blob = bucket.blob(blob_path)
    local_file_path = temp_file_path + image_filename

    blob.upload_from_filename(local_file_path)
    blob.make_public()  # Make the file publicly accessible
    return blob.public_url

def update_firestore_record(doc_id, predicted_class, predicted_probability, cam_image_url):
    """Updates the Firestore document with the processed image data."""
    doc_ref = firestore_db.collection("uploads").document(doc_id)
    # The following code updates the existing record with the new processed data fields
    doc_ref.update({
        "predicted_class": predicted_class,
        "predicted_probability": predicted_probability,
        "cam_image_url": cam_image_url,
        "processed_at": datetime.now(timezone.utc).isoformat()
    })
    print(f"Firestore document {doc_id} updated with processed data.")

def cleanup_images(filenames):
    """Removes the specified image files from the temporary directory."""
    for fname in filenames:
        file_path = temp_file_path + fname
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Temporary image file removed: {file_path}")
        else:
            print(f"File {file_path} does not exist and cannot be removed.")

if __name__ == "__main__":
    # Ensure the temp directory exists
    os.makedirs(temp_file_path, exist_ok=True)

    # If the script is run directly, process the image specified as an argument
    # Expecting doc_id as the first argument and image_filename as the second argument
    if len(sys.argv) > 2:
        doc_id = sys.argv[1]
        image_filename = sys.argv[2]
        process_image_prediction(doc_id, image_filename)
    else:
        print("Usage: python process_image.py <doc_id> <image_filename>")
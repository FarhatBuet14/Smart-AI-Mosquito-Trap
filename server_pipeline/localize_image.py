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

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
import os
import numpy as np
import cv2
import pandas as pd
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.structures import BoxMode

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ----------- Limiting GPU memory ----------------
tfConfig = tf.compat.v1.ConfigProto()
tfConfig.gpu_options.allow_growth = True
tfConfig.gpu_options.per_process_gpu_memory_fraction = 0.1  # allocating highest 10% GPU memory
sess = tf.compat.v1.Session(config=tfConfig)
tf.compat.v1.keras.backend.set_session(sess)


temp_file_path = "./temp_image_dir/"


# --- Load Annotations
anno_dir = "models/mosquito_localization/annotation"
train_df = pd.read_csv(f'{anno_dir}/train.csv')
val_df = pd.read_csv(f'{anno_dir}/val.csv')

classes = train_df.class_name.unique().tolist()

# Initialize Firebase Admin SDK and Firestore
SERVICE_ACCOUNT_KEY_PATH = 'score-trap-mqtt-test-firebase-adminsdk-wpt0n-1d8ecd74df.json'

if not firebase_admin._apps:
    cred = credentials.Certificate(SERVICE_ACCOUNT_KEY_PATH)
    firebase_admin.initialize_app(cred, {
        'storageBucket': 'score-trap-mqtt-test.appspot.com'
    })

firestore_db = firestore.client()
bucket = storage.bucket()

# def get_output_layer(model, layer_name):
#     layer_dict = dict([(layer.name, layer) for layer in model.layers])
#     layer = layer_dict[layer_name]
#     return layer

# --- Dataset Dictionary for Training
def create_dataset_dicts(df, classes):
    dataset_dicts = []
    for image_id, img_name in enumerate(df.file_name.unique()):
        record = {}
        image_df = df[df.file_name == img_name]
        record["file_name"] = list(df[df.file_name == img_name]["file_path"])[0]
        record["file_path"] = list(df[df.file_name == img_name]["file_path"])[0]
        record["image_id"] = image_id
        record["height"] = int(image_df.iloc[0].height)
        record["width"] = int(image_df.iloc[0].width)
        objs = []
        for _, row in image_df.iterrows():
            xmin = int(row.x_min)
            ymin = int(row.y_min)
            xmax = int(row.x_max)
            ymax = int(row.y_max)
            obj = {
            "bbox": [xmin, ymin, xmax, ymax],
            "bbox_mode": BoxMode.XYXY_ABS,
            "category_id": classes.index(row.class_name),
            "iscrowd": 0
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts

# --- Assign the Dictionary
for d in ["train", "val"]:
    DatasetCatalog.register("mark_" + d, lambda d=d: create_dataset_dicts(train_df if d == "train" else val_df, classes))
    MetadataCatalog.get("mark_" + d).set(thing_classes=classes)
statement_metadata = MetadataCatalog.get("mark_train")

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.num_gpus = 1
# cfg.MODEL.DEVICE = 'cuda'
cfg.MODEL.DEVICE = "cpu"
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.MODEL.MASK_ON = False

# --- Evaluation Setup
cfg.MODEL.WEIGHTS = os.path.join("models/mosquito_localization/model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.85
model = DefaultPredictor(cfg)

def execute(image_filename):
    
    im = cv2.imread(temp_file_path + image_filename)
    
    outputs = model(im)

    v = Visualizer(im[:, :, ::-1], metadata=statement_metadata, scale=1., instance_mode=ColorMode.IMAGE)
    instances = outputs["instances"].to("cpu")

    # Get boxes, scores and classes
    boxes = instances.pred_boxes.tensor.numpy()
    scores = instances.scores.numpy()

    # instances.remove('pred_masks')
    v = v.draw_instance_predictions(instances)
    result = v.get_image()[:, :, ::-1]
    
    # cam_filename with predicted class and probability
    base_name = os.path.splitext(image_filename)[0]
    cam_filename = f"{base_name}_localization.jpg"
    
    try:
        cv2.imwrite(temp_file_path + cam_filename, result)
    except Exception as e:
        print(f"Error writing to file: {temp_file_path}{cam_filename}, error: {e}")
    
    # Crop and save detected mosquitoes
    mos_crop_names = []
    for idx, (box, score) in enumerate(zip(boxes, scores)):
        if score < 0.85:
            continue
            
        # Convert box coordinates to integers
        x1, y1, x2, y2 = map(int, box)
        
        # Add padding around the box (10% of width/height)
        h, w = y2 - y1, x2 - x1
        pad_h = int(h * 0.1)
        pad_w = int(w * 0.1)
        
        # Adjust coordinates with padding, ensuring they stay within image bounds
        x1 = max(0, x1 - pad_w)
        y1 = max(0, y1 - pad_h)
        x2 = min(im.shape[1], x2 + pad_w)
        y2 = min(im.shape[0], y2 + pad_h)
        
        # Crop the mosquito
        crop = im[y1:y2, x1:x2]
        
        # Save the cropped image
        mos_name = f"{base_name}_mosquito_{idx+1}.jpg"
        
        try:
            cv2.imwrite(temp_file_path + mos_name, crop)
        except Exception as e:
            print(f"Error writing to file: {temp_file_path}{mos_name}, error: {e}")
        
        mos_crop_names.append(mos_name)

    return cam_filename, mos_crop_names

def process_image_prediction(doc_id, image_filename):
    print("\n\nNew image received: ", image_filename)

    # Run prediction and CAM generation
    print("Running prediction ....")
    cam_filename, mos_crop_names = execute(image_filename)

    base_name = os.path.splitext(image_filename)[0]

    # Dictionary to store all uploaded image links
    localized_images = {}

    # Upload localization image to Firebase Storage
    cam_image_url = upload_image_to_storage(cam_filename, f"mosquito_localization/{base_name}/")
    localized_images["localization_image"] = cam_image_url
    print(f"Uploaded localization image to Firebase Storage: {cam_image_url}")
    
    # Upload extracted mosquito images
    mos_urls = []
    for idx, mos in enumerate(mos_crop_names):
        mos_url = upload_image_to_storage(mos, f"mosquito_localization/{base_name}/")
        mos_urls.append(mos_url)
        localized_images[f"extracted_mosquito_{idx+1}"] = mos_url

    print(f"Uploaded separate mosquito images to Firebase Storage")

    # Update the Firestore document with processed data
    update_firestore_record(doc_id, localized_images)

    # Remove temporary images
    cleanup_images([cam_filename] + mos_crop_names)

    print("Image processing completed.")

def upload_image_to_storage(image_filename, folder_path):
    """Uploads the image file to Firebase Storage and returns its public URL."""
    blob_path = folder_path + image_filename
    blob = bucket.blob(blob_path)
    local_file_path = temp_file_path + image_filename

    blob.upload_from_filename(local_file_path)
    blob.make_public()  # Make the file publicly accessible
    return blob.public_url

def update_firestore_record(doc_id, localized_images):
    """Updates Firestore document with the `localizedImages` map containing processed image URLs."""
    doc_ref = firestore_db.collection("uploads").document(doc_id)
    
    # Update Firestore document
    doc_ref.update({
        "localizedImages": localized_images,
        "localized_at": datetime.now(timezone.utc).isoformat()
    })

    print(f"Firestore document {doc_id} updated with localized images.")

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
import os
import json
import numpy as np
import cv2
import tensorflow as tf
import tensorflow_hub as hub

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


IMG_DIR = "val2017_subset_regular_approach"     
ANNOT_FILE = "person_keypoints_val2017.json"  
MAX_SAMPLES = 973
OUTPUT_JSON = "movenet_pose_results.json"

MODEL_HANDLE = "https://tfhub.dev/google/movenet/singlepose/lightning/4"
# -------------------------

def load_img_local(img_path):
    bgr = cv2.imread(img_path)
    if bgr is None:
        raise ValueError(f"Failed to load image: {img_path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def run_movenet_and_eval():
    # Load COCO ground truth to filter valid images
    coco_gt = COCO(ANNOT_FILE)
    valid_ids = set(coco_gt.getImgIds(catIds=[1]))  # only images with 'person'

    # Collect image files that exist in ground truth
    all_imgs = [f for f in os.listdir(IMG_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    valid_imgs = [f for f in all_imgs if int(os.path.splitext(f)[0]) in valid_ids]

    n = min(len(valid_imgs), MAX_SAMPLES)
    subset = valid_imgs[:n]

    print(f"Processing {n} images from {IMG_DIR} with MoveNet (CPU)")

    # Load MoveNet
    print("Loading model from TensorFlow Hub:", MODEL_HANDLE)
    model = hub.load(MODEL_HANDLE)

    results = []
    for file_name in subset:
        img_path = os.path.join(IMG_DIR, file_name)
        try:
            img_rgb = load_img_local(img_path)
        except Exception as e:
            print("Skip sample, can't load image:", e)
            continue

        input_size = 192  # for singlepose lightning
        img_resized = tf.image.resize_with_pad(img_rgb, input_size, input_size)
        input_tensor = tf.expand_dims(img_resized, axis=0)
        input_tensor = tf.cast(input_tensor, dtype=tf.int32)

        outputs = model.signatures["serving_default"](input_tensor)
        keypoints_with_scores = outputs["output_0"].numpy()  # (1,1,17,3)

        h, w = img_rgb.shape[:2]
        kps = keypoints_with_scores[0, 0, :, :]  # (17,3)

        # Flatten keypoints as COCO format [x1,y1,v1,...]
        coco_keypoints = []
        for (y_norm, x_norm, score) in kps:
            x = float(x_norm * w)
            y = float(y_norm * h)
            if score > 0.5:
                v = 2  # visible
            elif score > 0.2:
                v = 1  # labeled but not visible
            else:
                v = 0  # not labeled
            coco_keypoints.extend([x, y, v])

        # COCO requires correct image_id
        image_id = int(os.path.splitext(file_name)[0])

        results.append({
            "image_id": image_id,
            "category_id": 1,  # person
            "keypoints": coco_keypoints,
            "score": float(np.mean(kps[:, 2]))  # average confidence
        })

    # Save predictions
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Saved {len(results)} results to {OUTPUT_JSON}")

    # ---- COCO Evaluation ----
    coco_dt = coco_gt.loadRes(OUTPUT_JSON)
    coco_eval = COCOeval(coco_gt, coco_dt, "keypoints")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()





if __name__ == "__main__":
    run_movenet_and_eval()

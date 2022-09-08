from pathlib import Path

from sahi.model import Yolov5DetectionModel
from sahi.utils.cv import visualize_object_predictions, read_image
from sahi.predict import get_prediction, get_sliced_prediction, predict
import numpy as np
import pickle
from models.experimental import attempt_load

from sahi.scripts.coco_error_analysis import analyse
from sahi.scripts.coco_evaluation import evaluate

MODEL_PATH = "trained_models/tood/latest.pth"
MODEL_CONFIG_PATH = 'trained_models/mmdet_configs/xview_tood/tood_crop_300_500_cls_cars_trucks_1e-3_new_pipe_csg_machine.py'
EVAL_IMAGES_FOLDER_DIR = '/vol/bitbucket/jrb21/project/xView/data/train/images'
EVAL_DATASET_JSON_PATH = "/vol/bitbucket/jrb21/project/xView/data/coco/val_cars_trucks.json"
INFERENCE_SETTING = 'XVIEW_SAHI_PO'
EXPORT_VISUAL = False

yolov5_model_path = 'trained_models/yolov5/best.pt'
model_type = "yolov5"
model_path = yolov5_model_path
model_device = "cuda:0"
model_confidence_threshold = 0.2

############ dont change below #############

INFERENCE_SETTING_TO_PARAMS = {
    "XVIEW_SAHI_PO": {
        "no_standard_prediction": True,
        "no_sliced_prediction": False,
        "slice_size": 400,
        "overlap_ratio": 0.20,
    }
}


setting_params = INFERENCE_SETTING_TO_PARAMS[INFERENCE_SETTING]

result = predict(
    model_type=model_type,
    model_path=model_path,
    model_confidence_threshold=0.01,
    model_device="cuda:0",
    model_category_mapping=None,
    model_category_remapping=None,
    source=EVAL_IMAGES_FOLDER_DIR,
    no_standard_prediction=setting_params["no_standard_prediction"],
    no_sliced_prediction=setting_params["no_sliced_prediction"],
    image_size=1280,
    slice_height=setting_params["slice_size"],
    slice_width=setting_params["slice_size"],
    overlap_height_ratio=setting_params["overlap_ratio"],
    overlap_width_ratio=setting_params["overlap_ratio"],
    postprocess_type="NMS",
    postprocess_match_metric="IOU",
    postprocess_match_threshold=0.5,
    postprocess_class_agnostic=False,
    novisual=not EXPORT_VISUAL,
    dataset_json_path=EVAL_DATASET_JSON_PATH,
    project="runs/predict_eval_analyse",
    name=INFERENCE_SETTING,
    visual_bbox_thickness=None,
    visual_text_size=None,
    visual_text_thickness=None,
    visual_export_format="png",
    verbose=0,
    return_dict=True,
    force_postprocess_type=True,
)

result_json_path = str(Path(result["export_dir"]) / "result.json")

evaluate(
    dataset_json_path=EVAL_DATASET_JSON_PATH,
    result_json_path=result_json_path,
    classwise=True,
    max_detections=500,
    return_dict=False,
)

analyse(
    dataset_json_path=EVAL_DATASET_JSON_PATH,
    result_json_path=result_json_path,
    max_detections=500,
    return_dict=False,
)

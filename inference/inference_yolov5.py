# import required functions, classes
from sahi.model import Yolov5DetectionModel
from sahi.utils.cv import visualize_object_predictions, read_image
from sahi.predict import get_prediction, get_sliced_prediction, predict
import numpy as np
import pickle
from models.experimental import attempt_load
print('imports success')


yolov5_model_path = 'trained_models/yolov5/best.pt'
model_type = "yolov5"
model_path = yolov5_model_path
model_device = "cuda:0"
model_confidence_threshold = 0.2

slice_height = 400
slice_width = 400
overlap_height_ratio = 0.2
overlap_width_ratio = 0.2

source_image_dir = "WV3/"

INFERENCE_SETTING_TO_PARAMS = {
    "XVIEW_SAHI_PO": {
        "no_standard_prediction": True,
        "no_sliced_prediction": False,
        "slice_size": 400,
        "overlap_ratio": 0.20,
    }
}

# Of the above 4 options I want to slice and not do full image size inference
INFERENCE_SETTING = "XVIEW_SAHI_PO"
setting_params = INFERENCE_SETTING_TO_PARAMS[INFERENCE_SETTING]

# From the evaluation py file I delete the eval dataset path and change the confidence to 0.3
result = predict(
    model_type=model_type,
    model_path=model_path,
    model_confidence_threshold=model_confidence_threshold,
    model_device=model_device,
    model_category_mapping=None,
    model_category_remapping=None,
    source=source_image_dir,
    no_standard_prediction=setting_params["no_standard_prediction"],
    no_sliced_prediction=setting_params["no_sliced_prediction"],
    image_size=None,
    slice_height=setting_params["slice_size"],
    slice_width=setting_params["slice_size"],
    overlap_height_ratio=setting_params["overlap_ratio"],
    overlap_width_ratio=setting_params["overlap_ratio"],
    postprocess_type="GREEDYNMM",
    postprocess_match_metric="IOS",
    postprocess_match_threshold=0.5,
    postprocess_class_agnostic=True,
    visual_bbox_thickness=1,
    visual_text_size=0.3,
    visual_text_thickness=1,
    visual_export_format="png",
    verbose=0,
    return_dict=True,
    force_postprocess_type=True
)
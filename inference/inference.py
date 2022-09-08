import argparse
from sahi.model import Yolov5DetectionModel
from sahi.utils.cv import visualize_object_predictions, read_image
from sahi.predict import get_prediction, get_sliced_prediction, predict
import numpy as np
import pickle
from models.experimental import attempt_load


parser = argparse.ArgumentParser(description='Model the traffic flow')
parser.add_argument('--model_path', type=str, default='trained_models/yolov5/best.pt')
parser.add_argument('--source_image_dir', type=str, default='WV3/')
parser.add_argument('--name', type=str, default='yolov5')
parser.add_argument('--export_pickle', nargs='?', const=True, default=False)
parser.add_argument('--novisual', nargs='?', const=True, default=False)
parser.add_argument('--conf_thresh', type=float, default=0.15)
args = parser.parse_args()


print(args.conf_thresh)

model_type = "yolov5"
model_device = "cuda:0"
slice_size = 400
overlap_ratio = 0.20


result = predict(
    model_type=model_type,
    model_path=args.model_path,
    model_confidence_threshold=args.conf_thresh,
    model_device=model_device,
    model_category_mapping=None,
    model_category_remapping=None,
    source=args.source_image_dir,
    no_standard_prediction=True,
    no_sliced_prediction=False,
    image_size=None,
    slice_height=slice_size,
    slice_width=slice_size,
    overlap_height_ratio=overlap_ratio,
    overlap_width_ratio=overlap_ratio,
    postprocess_type="GREEDYNMM",
    postprocess_match_metric="IOS",
    postprocess_match_threshold=0.5,
    postprocess_class_agnostic=True,
    visual_bbox_thickness=1,
    visual_text_size=0.1,
    visual_text_thickness=1,
    visual_export_format="png",
    verbose=0,
    return_dict=True,
    force_postprocess_type=True,
    project = "results",
    name = args.name,
    export_pickle = args.export_pickle,
    novisual = args.novisual
)
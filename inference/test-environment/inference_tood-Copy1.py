from sahi.model import MmdetDetectionModel
from sahi.utils.cv import visualize_object_predictions, read_image
from sahi.predict import get_prediction, get_sliced_prediction, predict
import numpy as np
import pickle
print('import success')


tood_config_path = 'trained_models/mmdet_configs/xview_tood/tood_crop_300_500_cls_cars_trucks_1e-3_new_pipe_csg_machine.py'
tood_model_path = "trained_models/tood/latest.pth"

detection_model = MmdetDetectionModel(
    model_path= tood_model_path,
    config_path= tood_config_path,
    device='cuda:0' # or 'cpu'
)


print('detection model success')
# import required functions, classes
from sahi.model import Yolov5DetectionModel
from sahi.utils.cv import visualize_object_predictions, read_image
from sahi.predict import get_prediction, get_sliced_prediction, predict
import numpy as np
import pickle
print('imports success')
from models.experimental import attempt_load
print('second imports success')


yolov5_model_path = 'trained_models/yolov5/best.pt'

detection_model = Yolov5DetectionModel(
    model_path=yolov5_model_path,
    confidence_threshold=0.3,
    device="cuda:0", # or 'cuda:0'
)

print('sucessfully loaded model')
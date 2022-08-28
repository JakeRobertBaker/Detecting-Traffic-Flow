from sahi.slicing import slice_coco
from sahi.utils.file import load_json
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from collections import defaultdict
import copy
import random


# Slice the training set
coco_dict, coco_path = slice_coco(
    coco_annotation_file_path="/rds/general/user/jrb21/home/small-object-detection-benchmark/data/xview/coco/train_cars_trucks.json",
    image_dir="/rds/general/user/jrb21/home/train/train_images/",
    output_coco_annotation_file_name="slice_train_images",
    ignore_negative_samples=False,
    output_dir="/rds/general/user/jrb21/home/train/slice_train_images_with_negative/",
    slice_height=400,
    slice_width=400,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
    min_area_ratio=0.1,
    verbose=False,
    out_ext = '.png'
)

# Open the training json
with open("/rds/general/user/jrb21/home/train/slice_train_images_with_negative/slice_train_images_coco.json") as train_file:
    data_train = json.load(train_file)
    
# Redistriubute the dataset
def redistribute(data):
    annotation_count = defaultdict(int)
    for ann in data['annotations']:
        image_id = ann['image_id']
        annotation_count[image_id]+=1
    
    
    more_than_four = {k: v for k, v in annotation_count.items() if v >= 4}
    three = {k: v for k, v in annotation_count.items() if v == 3}
    two = {k: v for k, v in annotation_count.items() if v == 2}
    one = {k: v for k, v in annotation_count.items() if v == 1}
    zero = {k: v for k, v in annotation_count.items() if v == 0}
    
    length_more_than_four = len(more_than_four.values())
    length_all = int(length_more_than_four/0.9)
    length_zero = int(0.07 * length_all)
    length_one = int(0.01 * length_all)
    length_two = int(0.01 * length_all)
    length_three = int(0.01 * length_all)
    
    
    more_than_four_ids = list(more_than_four.keys())
    three_ids = random.sample(list(three.keys()), length_three)
    two_ids = random.sample(list(two.keys()), length_two)
    one_ids = random.sample(list(one.keys()), length_one)
    
    ids = more_than_four_ids + three_ids + two_ids + one_ids
    
    # Find images with at least n annotations    
    images = []
    for img in data['images']:
        image_id = img['id']
        if image_id in ids:
            images.append(img)
            
    # find annotations with at least n annotations
    annotations = []
    for ann in data['annotations']:
        image_id = ann['image_id']
        if image_id in ids:
            annotations.append(ann)
            
    new_data = copy.copy(data)
    new_data['images'] = images
    new_data['annotations'] = annotations
    
    return new_data
  
  
new_data_train = redistribute(data_train)

# Save output
with open('/rds/general/user/jrb21/home/train/slice_json/trimmed_train_include_negative.json', 'w') as f:
    json.dump(new_data_train, f)
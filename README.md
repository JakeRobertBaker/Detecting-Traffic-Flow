# Detecting Traffic Flows

## Getting started

1. This repo contains two submodules, yolov5 [1] and sahi [2]. After cloning this repo run `git submodule update` to download the and update these repos.

[1] https://github.com/ultralytics/yolov5

[2] https://github.com/obss/sahi

2. Make sure the symlinks pointing between folders are there. If not run 

```bash
cd inference 
ln -s ../sahi/sahi sahi
ln -s ../yolov5/models models
ln -s ../yolov5/utils utils
```

3. Download yolov5 weights to `inference/trained_models/yolov5/` and tood weights to `inference/trained_models/tood/`.

4. Environment setup:
 
    Inference:
    ```bash
    conda create --name inference
    conda activate inference
    conda install pytorch=1.10.0 torchvision=0.11.1 cudatoolkit=11.3 -c pytorch
    cd inference
    pip install -r trained_models/tood/tood-req.txt
    pip install -r ../sahi/requirements.txt
    pip install yolov5
    ```
    
    tf:
    ```
    11.2.1-cudnn8.1.0.77
    source /vol/cuda/11.2.1-cudnn8.1.0.77/setup.sh

    ```
    
    
    
## Inference

To detect traffic on images make sure to have the images inside `inference/WV3` and then run `cd inference` and `python inference_yolov5.py` or `python inference_tood.py` for the (worse peforming) tood detector.

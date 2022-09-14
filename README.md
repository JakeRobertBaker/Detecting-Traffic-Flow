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


To detect traffic run without image exports
```bash

python inference.py --conf_thresh 0.05 --novisual --export_pickle --name single-lane-yolov5
python inference.py --conf_thresh 0.05 --novisual --export_pickle --source_image_dir WV3-dual-lane --name double-lane-yolov5
```

## Traffic Count Prediction

Once inference on the images has been peformed Traffic Count predictions can be made.
Please see the notebook Traffic-Count-Pipeline in the folder traffic-flow for a step by step walkthrough of the inference process. Before running the notebook Traffic Count data needs downloading. The files download_high_quality_reports.py, download_image_time_reports.py, download_image_year_reports.py must be run.

## AADT predictions
The linear model needs around 100GB of RAM to run, the dataset is huge! To produce the model run Flow-Model-Baseline.py.
Afterwards the notebooks Flow-Model and Flow-Model-Linear walkthrough the prediction of AADT/AMD with median and linear models respectively.

## Plots
This is a miscellanious notebook used to create plots for my report. I have left it in for the curious to explore.

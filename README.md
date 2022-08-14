# Traffic Light Detection and Mapping for HD-map Creation

<img src="https://github.com/yusiyohpolimi/thesis/diagram_new.jpg" height="480">

The thesis is developed by Yusuf Can Simsek. Note that this repository is designed to work in Artificial Intelligence and Robotics Laboratory at Politecnico di Milano (AIRLab) servers.

## Installation

Docker environment
<details><summary> <b>Expand</b> </summary>

```
# Build the docker container which has the all components to run OSRM service, YOLO models and detectron2 models
docker build --rm -t <tag> -f Venv .  
```

</details>

Clone the repository and download the [model weights](https://polimi365-my.sharepoint.com/personal/10622973_polimi_it/_layouts/15/onedrive.aspx?id=%2Fpersonal%2F10622973_polimi_it%2FDocuments%2FTesi%2Ftesi%20Yusuf%2Fmodel_weights&ga=1) into the base folder. Then run bash script for the weights:
```
git clone https://github.com/yusiyohpolimi/thesis.git

cd traffic_lights_detection/
# Download the weights here and run following script
bash weights.sh
```
  
Map installation for OSRM service
<details><summary> <b>Expand</b> </summary>

```
cd osrm/
# To download all the germany map:
wget http://download.geofabrik.de/europe/germany-latest.osm.pbf

# For just Berlin:
wget http://download.geofabrik.de/europe/germany/berlin-latest.osm.pbf


# Extract the downloaded map:
run-docker --image_name="simsek:thesis_last" '' '' osrm-extract -p /osrm-backend/profiles/car.lua germany-latest.osm.pbf

# Partition:
run-docker --image_name="simsek:thesis_last" '' '' osrm-partition germany-latest.osrm

# Customize:
run-docker --image_name="simsek:thesis_last" '' '' osrm-customize germany-latest.osrm 

# Do not forget to change the tag if you need
```

</details>
  
To train/test YOLO models, inputs should be in YOLO format. Check [yolo_scripts](https://github.com/AIRLab-POLIMI/traffic_lights_detection/tree/main/yolo_scripts) for different cases. Note that the path for dataset is hardcoded. For multi-label task, use:
```
python3 yolo_format_bulb.py
```  

## Usage of Different Models

CSP models (CSP10px, CSP640, CSP640multi):
<details><summary> <b>Expand</b> </summary>  

```
train.py	
          --weights <path of the initial weights> 
          --cfg <model.yaml/.cfg path>
          --data <data.yaml path>
          --epochs
          --batch-size
          --img-size <single int input, e.g. 640>
          --rect <rectangular training option for non square input sizes>
          --resume <resuming from given weight>
          --logdir <logging directory>
          # check help for other arguments

test.py		
          --weights 
          --data 
          --batch-size
          --img-size
          --conf-thres <object confidence score threshold>
          --iou-thres <IOU threshold for NMS>
          --task <'val', 'test'>
          --verbose <report results by class in bash>
          # check help for other arguments

detect.py 	
            --weights
            --source <path of the input json file with image paths and vehicle data>
            --output <folder to save detection outputs>
            --img-size
            --conf-thres
            --iou-thres
            # check help for other arguments
```
</details>
  
  
YOLOR models:
<details><summary> <b>Expand</b> </summary>  

```
train.py	
          --weights <path of the initial weights> 
          --cfg <model.yaml/.cfg path>
          --data <data.yaml path>
          --epochs
          --batch-size
          --img-size <single int input, e.g. 1280>
          --rect <rectangular training option for non square input sizes>
          --resume <resuming from given weight>
          --name <works like logdir in CSP models>		
          # check help for other arguments

test.py		
          --weights 
          --data 
          --batch-size
          --img-size
          --conf-thres <object confidence score threshold>
          --iou-thres <IOU threshold for NMS>
          --task <'val', 'test'>
          --verbose <report results by class in bash>
          --name <save folder for the results and plots>
          # check help for other arguments

detect.py 	
          --weights
          --source <path of the input json file with image paths and vehicle data>
          --name <folder to save detection outputs>
          --img-size
          --conf-thres
          --iou-thres
          # check help for other arguments
```
</details>
  
  
Detectron2 models:
<details><summary> <b>Expand</b> </summary>  

```
train.py	
          --weights <path of the initial weights> 
          --data <json file of the trainset>
          --epochs
          --batch-size
          --img-size <single int input, e.g. 1024 2048>	
          --resume <resuming from given weight>
          --logdir <works like logdir in CSP models>
          --anchors <using custom anchors generated>	
          # check help for other arguments

test.py		
          --weights 
          --data <json file of the validation/test set, use train json for validaiton task>
          --img-size
          --conf <object confidence score threshold>
          --iou <IOU threshold for NMS>
          --task <'val', 'test'>
          --logdir <save folder for the results and plots>
          --ratio <ratio of the test set to be used>
          --anchors
          # check help for other arguments

detect.py 	
          --weights
          --json-path <path of the input json file with image paths and vehicle data>
          --output <folder to save detection outputs>
          --img-size
          --conf
          --iou
          --anchors
          # check help for other arguments
```
</details>
 
## Example Commands for Test and Detection

CSP640multi:
``` 
# test
run-docker --image_name="simsek:thesis_last" <gpu_number> <cpu_number> python3 csp/test.py --weights csp/weights/csp640multi/weights/best.pt --img 640 --data csp/data/dtld_bulb_100.yaml --iou 0.5 --conf 0.4 --batch 112 --task test --verbose

# detect
run-docker --image_name="simsek:thesis_last" <gpu_number> <cpu_number> python3 csp/detect.py --weights ./csp/weights/csp640multi/weights/best.pt --img 640 --source berlin1_inp.json --iou 0.5 --conf 0.4 --output <output dir name>
```

YOLOR-D6multi:
``` 
# test
run-docker --image_name="simsek:thesis_last" <gpu_number> <cpu_number> python3 yolor/test.py --weights yolor/weights/YOLOR-D6multi/weights/best.pt --img 640 --data yolor/data/ dtld_bulb_100.yaml --iou 0.5 --conf 0.4 --batch 12 --task test --verbose

# detect
run-docker --image_name="simsek:thesis_last" <gpu_number> <cpu_number> python3 yolor/detect.py --weights yolor/weights/YOLOR-D6multi/weights/best.pt --img 2048 --source berlin1_inp.json --iou 0.5 --conf 0.4 --name <output dir name>
```

Detectron2:
```
# test
run-docker --image_name="simsek:thesis_last" <gpu_number> <cpu_number> python3 detectron2/test.py --weights detectron2/weights/R50_30_130/model_best_ap50.pth --iou 0.0 --img 1024 2048 --conf 0.8 --data /multiverse/datasets/shared/DTLD/v2.0/v2.0/DTLD_test.json --anchors

# detect
run-docker --image_name="simsek:thesis_last" <gpu_number> <cpu_number> python3 detectron2/detect.py --weights detectron/weights/R50_30_130/model_best_ap50.pth --img 1024 2048 --json berlin1_inp.json --iou 0.5 --conf 0.4 --output <output dir name> --anchors
```

## Generating JSON file for Detection on DTLD

Using the default_input.py script, given json file is cleaned such that there is no ground-truths. Then, new json file is generated to be used as input for the models.
```
python3 default_input.py
```

## Running the Pipeline

To run the full pipeline, use the bash script which first opens the OSRM service, and then starts pipeline.py which runs detection and mapping:
```
run-docker --image_name="simsek:thesis_last" <gpu_number> <cpu_number> bash start.sh
# Check the start.sh for the given arguments of the pipeline.py.
```

It is also possible to run mapping on the detection json file:
```
run-docker --image_name="simsek:thesis_last" <gpu_number> <cpu_number> bash onlymapping.sh
# Check the onlymapping.sh for the given arguments of the mapping.py
```

## Notes
 
*	Building the docker image and extracting the map for OSRM takes a long-time. Both are needed for once.
*	While running, snapping takes the most of the time after detection. Better GPS measurements will eliminate this.
*	Disparity images only for Berlin are extracted since they consume quite a big space: 
  /multiverse/datasets/shared/DTLD/Berlin_disp for Berlin1, /multiverse/datasets/shared/DTLD/Berlin_disp_full for all Berlin sequences.
*	There are some hard coded paths for dataset location in multiverse, which is: 
  /multiverse/datasets/shared/DTLD/
*	Note that image_name should be the tag of the docker created in the first step.


 

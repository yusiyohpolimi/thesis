#!/bin/bash
  
# turn on bash's job control
set -m
  
# Start OSRM server to snap the GPS measurements
osrm-routed --algorithm mld osrm/germany-latest.osrm --max-matching-size 120 &
  
# Start the detection and mapping script
python3 pipeline.py \
    --weights det2/weights/R50_30_130/model_best_ap50.pth \
    --iou 0.0 --img 1024 2048 --conf 0.3 --output out_new --anchors \
    --json-path berlin1_inp.json
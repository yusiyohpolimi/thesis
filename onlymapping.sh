#!/bin/bash
  
# turn on bash's job control
set -m
  
# Start OSRM server to snap the GPS measurements
osrm-routed --algorithm mld osrm/germany-latest.osrm --max-matching-size 120 &

sleep 30

# Start the detection and mapping script
python3 mapping.py --database-dir /multiverse/datasets/shared/DTLD/Berlin_disp \
--label yolor/2022_04_16_2232_berlin1_inp_detection.json

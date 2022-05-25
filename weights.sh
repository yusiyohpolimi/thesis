#!/bin/bash
set -m

unzip all_weights.zip -d temp &&

mkdir -p csp/weights/ &&
mkdir -p yolor/weights/ &&
mkdir -p detectron2/weights/ &&

mv temp/all_weights/csp* csp/weights/ &&
mv temp/all_weights/YOLOR* yolor/weights/ &&
mv temp/all_weights/R50* detectron2/weights/ &&
rm -r temp/


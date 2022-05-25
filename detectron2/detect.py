import torch, os, argparse, copy
import numpy as np
import detectron2.modeling.roi_heads.fast_rcnn as fast 
import glob

from detectron2 import model_zoo
from detectron2.utils.logger import setup_logger
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import detection_utils as utils
from general import *


def main(opt):
    fast.FastRCNNOutputLayers.predict_probs = predict_probs2
    utils.read_image = read_image2
    utils.annotations_to_instances = anno_to_instances

    cfg = get_cfg()
    if opt.model == 'R50':
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    elif opt.model == 'X101':
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 7

    try:
        assert opt.weights != '', "Provide initial weights for model"
    except AssertionError:
        raise        

    cfg_path = '/'.join(opt.weights.split('/')[:-1]) + '/'   
    cfg.merge_from_file(os.path.join(cfg_path, "config.yaml"))
    cfg.MODEL.WEIGHTS = opt.weights
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = opt.conf    # set the testing threshold for this model
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = opt.iou
    if opt.anchors:
        cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[8], [12], [21], [32], [64], [128], [256], [512]]
        cfg.MODEL.RPN.IN_FEATURES = ['p2', 'p2', 'p2', 'p2', 'p3', 'p4', 'p5', 'p6']
        cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[3.25]]    
 
    if opt.cpu:
        cfg.MODEL.DEVICE='cpu'
  
    cfg.INPUT.MIN_SIZE_TEST = 0

    setup_logger(output=cfg.OUTPUT_DIR)
    logger = logging.getLogger('detectron2')
    # logger.info(cfg)
    det2_path = 'det2/' 

    cfg.OUTPUT_DIR = os.path.join(det2_path, opt.output)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    predictor = DefaultPredictor(cfg)    
    # img_list = glob.glob(os.path.join(opt.source, '*k0.tiff')) 
    
    import time
    start = time.time()
    json_path = opt.json_path
    with open(json_path) as f:
        parsed = json.load(f)
    images = parsed['images']
    out_list_json = []

    for image in images:        
        img_path = image['image_path']
        path_split = img_path.split('/')
        # This image is problematic and cannot be read by opencv2, thus omit it
        # if path_split[-1] == 'DE_BBBR667_2015-04-23_13-20-52-402607_k0.tiff':
        #     continue        
        if path_split[0] != '/multiverse/':
            path_split[0] = '/multiverse/datasets/shared/DTLD' # path in the westworld/multiverse server
        im_path = '/'.join(path_split)
        im = read_image2(im_path)
        out = predictor(im)
        out_list_json.append(out)
        out = np.array([out, im_path])
        draw_labels(out=out, save_path=cfg.OUTPUT_DIR)      
        print('{} done'.format(im_path))          
    passed = time.time() - start
    print('It is done in {} s'.format(passed))
    label_file = save_detect(json_path, out_list_json, 'det2')

    return label_file


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='', help='initial weights path')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='test sizes')
    parser.add_argument('--iou', type=float, default=0.5, help='nms iou threshold')
    parser.add_argument('--conf', type=float, default=0.5, help='confidence score threshold')
    parser.add_argument('--anchors', action='store_true', help='use custom anchors')
    parser.add_argument('--model', type=str, default='R50', help='R50, X101')
    parser.add_argument('--json-path', type=str, default='/multiverse/datasets/shared/DTLD/v2.0/v2.0/default_inp.json', 
                        help='input json file with image paths and vehicle data')
    parser.add_argument('--cpu', action='store_true', help='use cpu for inference')
    parser.add_argument('--output', type=str, default='', help='folder to save detections')
      
    opt = parser.parse_args()
    
    main(opt)







import torch, os, argparse, copy
import numpy as np
import detectron2.modeling.roi_heads.fast_rcnn as fast 
from detectron2 import model_zoo
from detectron2.utils.logger import setup_logger
from detectron2.config import get_cfg
from detectron2.data import transforms as T
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.dataset_mapper import *
from detectron2.data import detection_utils as utils
from general import *


def __callpredaug__(self, original_image):
    """
    This is the custom call method for DefaultPredictor class. 
    It is used for prediction by using the class directly for 
    one image, instead of using inference_on_dataset function.
    """
    with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
        # Apply pre-processing to image.
        if self.input_format == "RGB":
            # whether the model expects BGR inputs or RGB
            original_image = original_image[:, :, ::-1]
        height, width = original_image.shape[:2]
        self.aug = T.Resize((640,640)) # change the image to 640x640 for prediction single image
        image = self.aug.get_transform(original_image).apply_image(original_image)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        print(image.shape, height, width)        
        inputs = {"image": image, "height": height, "width": width}
        predictions = self.model([inputs])[0]
        return predictions


def main(opt):
    DefaultPredictor.__call__ =  __callpredaug__
    fast.FastRCNNOutputLayers.predict_probs = predict_probs2
    utils.read_image = read_image2
    utils.annotations_to_instances = anno_to_instances

    json_path = opt.data
    if opt.task == 'val':
        print('\nValidating and calculating the metrics for the validation set from DTLD...')     
        _, X_val = DTLD_splitter(json_path, opt.task)
        data_size = len(X_val)
        data_used = round(data_size * opt.ratio)
        DatasetCatalog.register("val_set", lambda d=X_val[:data_used]: get_DTLD(d))
        MetadataCatalog.get("val_set").set(thing_classes=['1B', '2B', '3B', 'F', 'B', 'R', 'L'])
        dataset_name = 'val_set'
        
    elif opt.task == 'test':
        json_path = '/multiverse/datasets/shared/DTLD/v2.0/v2.0/DTLD_test.json'
        # json_path = '/multiverse/datasets/shared/DTLD/detectron2/full_size_mini/mini_3.json'
        print('\nTesting and calculating the metrics for the testset from DTLD...')  
        _, X_val = DTLD_splitter(json_path, opt.task)
        data_size = len(X_val)
        data_used = round(data_size * opt.ratio)
        DatasetCatalog.register("test_set", lambda d=X_val[:data_used]: get_DTLD(d))
        MetadataCatalog.get("test_set").set(thing_classes=['1B', '2B', '3B', 'F', 'B', 'R', 'L'])
        dataset_name = 'test_set'

    cfg = get_cfg()
    if opt.model == 'R50':
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    elif opt.model == 'X101':
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 7
    det2_path = 'det2' 
    cfg.OUTPUT_DIR = os.path.join(det2_path, opt.logdir)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    try:
        assert opt.weights != '', "Provide initial weights for model"
    except AssertionError:
        raise        

    cfg_path = '/'.join(opt.weights.split('/')[:-1]) + '/'   
    cfg.merge_from_file(os.path.join(cfg_path, "config.yaml"))
    cfg.MODEL.WEIGHTS = opt.weights
    cfg.DATASETS.TEST = (dataset_name, )
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = opt.conf    # set the testing threshold for this model
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = opt.iou
    cfg.CUDNN_BENCHMARK = True
    if opt.anchors:
        cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[8], [12], [21], [32], [64], [128], [256], [512]]
        cfg.MODEL.RPN.IN_FEATURES = ['p2', 'p2', 'p2', 'p2', 'p3', 'p4', 'p5', 'p6']
        cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[3.25]]    

    if opt.cpu:
        cfg.MODEL.DEVICE='cpu'
        
    setup_logger(output=cfg.OUTPUT_DIR)
    logger = logging.getLogger('detectron2')
    # logger.info(cfg)
    predictor = DefaultPredictor(cfg)

    if opt.im_path:
        print('\nPredicting for a single image provided...')
        im = read_image2(opt.im_path)
        out = predictor(im)
        print(out['instances'])
        draw_labels(out=out, im_path=opt.im_path)

    else:
        print('\nUsing inference_on_dataset function...')
        from detectron2.data import build_detection_test_loader
        from detectron2.evaluation.evaluator import inference_on_dataset
        val_loader = build_detection_test_loader(cfg, 
                                                dataset_name, 
                                                mapper=CustomMapper(cfg, is_train=False, 
                                                                    augmentations=[T.Resize((opt.img_size[0], 
                                                                                             opt.img_size[1]))]))
        evaluator = CustomEvaluator(dataset_name)
        out = inference_on_dataset(predictor.model, val_loader, evaluator)       
        if opt.save_detect:
            # test_folder = '/multiverse/datasets/shared/DTLD/testing/bulb_100' 
            for pred in out[1].reshape((-1, 2)):
                draw_labels(out=pred, save_path=cfg.OUTPUT_DIR)
        
        if opt.save_json:
            outputs = out[1].reshape((-1, 2))
            save_pred(json_path, outputs, save_path=cfg.OUTPUT_DIR)
            # return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='', help='initial weights path')
    parser.add_argument('--data', type=str, 
                        default='/multiverse/datasets/shared/DTLD/v2.0/v2.0/DTLD_train.json', 
                        help='json file of trainset')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='test sizes')
    parser.add_argument('--iou', type=float, default=0.5, help='nms iou threshold')
    parser.add_argument('--conf', type=float, default=0.5, help='confidence score threshold')
    parser.add_argument('--logdir', type=str, default='', help='logging directory')
    parser.add_argument('--im_path', type=str, default='', help='path for prediction on single image')
    parser.add_argument('--task', default='test', help="'val', 'test'")
    parser.add_argument('--ratio', type=float, default=1.0, help="ratio of the val/test set to use")
    parser.add_argument('--anchors', action='store_true', help='use custom anchors')
    parser.add_argument('--model', type=str, default='R50', help='R50, X101')
    parser.add_argument('--save-json', action='store_true', help='save results in DTLD json format')
    parser.add_argument('--save-detect', action='store_true', help='save detections in output folder')
    parser.add_argument('--cpu', action='store_true', help='use cpu for inference')
    
    opt = parser.parse_args()
    
    main(opt)







import logging
import copy
from re import L
import cv2
import torch
import matplotlib
import json
import os
import matplotlib.pyplot as plt
import numpy as np

import detectron2.data.detection_utils as util
import detectron2.modeling.roi_heads.fast_rcnn as fast   
from detectron2.utils.events import get_event_storage
from detectron2.structures import (
    Boxes,
    BoxMode,
    Instances,
)
from detectron2.data import transforms as T
from detectron2.evaluation.evaluator import *
from detectron2.structures import BoxMode
from detectron2.data import MetadataCatalog
from detectron2.config import get_cfg
from detectron2.data.dataset_mapper import *
from detectron2.data import detection_utils as utils

from collections import OrderedDict
from torchvision.ops import box_iou
from sklearn.model_selection import train_test_split
from typing import List, Tuple
from torch.nn import functional as F


def DTLD_splitter(json_path, task):
    """
    Split the DTLD into train and validation sets, 
    or return test set only if the task is testing.
    """
    with open(json_path) as f:
        parsed = json.load(f)

    images = parsed['images']
    if task == 'val' or task == 'train':
        percentage_test = 0.2
        X_train, X_val= train_test_split(images, test_size=percentage_test, random_state=31)
        return X_train, X_val 

    else:        
        X_val = images
        return [], X_val          


def get_DTLD(input_data, test_folder=None, dataset_dir='/multiverse/datasets/shared/DTLD'):
    """    
    Getting input data and returning a dictionary 
    in required format for detectron2 pipeline.
    """ 
    obj_count = 0
    dataset_dicts = []
    # test_folder = '/multiverse/datasets/shared/DTLD/testing/bulb_100'   #
    # test_folder = '/multiverse/datasets/shared/DTLD/Berlin_disp/Berlin/Berlin1/*' # 2015-04-17_11-07-02'   #
    import glob
    if test_folder:
        img_list = glob.glob(os.path.join(test_folder, '*k0.tiff'))     #

    for idx, image in enumerate(input_data):        
        record = {}
        img_path = image['image_path']
        path_split = img_path.split('/')
        # This image is problematic and cannot be read by opencv2, thus omit it
        if path_split[-1] == 'DE_BBBR667_2015-04-23_13-20-52-402607_k0.tiff':
            continue
        
        if test_folder:
            img_name = path_split[-1]
            if img_name not in [x.split('/')[-1] for x in img_list]:
                continue
        #
        path_split[0] = dataset_dir # path in the westworld/multiverse server
        filename = '/'.join(path_split)
        height, width = 1024, 2048       
        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width             
        objs = []   

        for anno in image['labels']:
            x = anno['x']
            y = anno['y']
            w = anno['w']
            h = anno['h']
            # Not allowing the bbox coordinates outside of the image
            if x < 0:
                w = w + x
                x = 0                       
            elif y < 0:
                h = h + y
                y = 0
            elif x + w > 2048:
                w = 2048 - x
            elif y + h > 1024:
                h = 1024 - y     

            xmin = x
            ymin = y
            xmax = xmin + w 
            ymax = ymin + h
            
            aspect = anno['attributes']['aspects']
            direction = anno['attributes']['direction']
            orientation = anno['attributes']['orientation']
            occ = anno['attributes']['occlusion']
            reflection = anno['attributes']['reflection']   
            # Skipping irrelevant attributes          
            if reflection == 'reflected' or occ == 'occluded' or aspect == 'four_aspects' or \
               orientation == 'horizontal' or aspect == 'unknown':
                continue            
            obj_count += 1 
            # Seven classes that are the focus of the study
            dict_7 = {'one_aspect': 0, 
                    'two_aspects': 1,
                    'three_aspects': 2,
                    'front': 3,
                    'back': 4,
                    'right': 5,
                    'left': 6}
            # Instead of giving single ID, a list of classes is given as the category ID
            # which is the suitable way to do multilabel detection
            # Note that default detectron2 pipeline cannot read multiple category ID
            id = [dict_7[aspect], dict_7[direction]]
            obj = {
                "bbox": [xmin, ymin, xmax, ymax],
                "bbox_mode": BoxMode.XYXY_ABS,                
                "category_id": id,
                }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)

    return dataset_dicts


def anno_to_instances(annos, image_size, mask_format="polygon"):  
    """
    Made changes to make the original function available for 
    multi-label tasks and cut off the unrelated part like segmentation/keypoint.       
    """
    boxes = (
        np.stack(
            [BoxMode.convert(obj["bbox"], obj["bbox_mode"], BoxMode.XYXY_ABS) for obj in annos]
        )
        if len(annos)
        else np.zeros((0, 4))
    )
    target = Instances(image_size)
    target.gt_boxes = Boxes(boxes)

    classes = [(obj["category_id"]) for obj in annos] # this makes the multi-label definition
    classes = torch.tensor(classes, dtype=torch.int64)
    target.gt_classes = classes

    return target


def transform_instance_annotations2(
    annotation, transforms, image_size, *, keypoint_hflip_indices=None
):
    """
    After resizing the images to smaller sizes, some of the widths of the labels become
    smaller than 1 pixel, which yields 0 when converted to integer. To overcome this issue,
    enforcing the boxes to have at least 1 pixel width/height. In addition, cut off the unrelated
    sections.
    """
    if isinstance(transforms, (tuple, list)):
        transforms = T.TransformList(transforms)
    # bbox is 1d (per-instance bounding box)
    bbox = BoxMode.convert(annotation["bbox"], annotation["bbox_mode"], BoxMode.XYXY_ABS)
    # clip transformed bbox to image size
    bbox = transforms.apply_box(np.array([bbox]))[0].clip(min=0)
    
    #Increasing the larger coordinate to have at least 1 pixel width/height
    if bbox[0] == bbox[2]:
        bbox[2] += 1
    elif bbox[1] == bbox[3]:
        bbox[3] += 1
        
    annotation["bbox"] = np.minimum(bbox, list(image_size + image_size)[::-1])
    annotation["bbox_mode"] = BoxMode.XYXY_ABS

    return annotation


def predict_probs2(
    self, predictions: Tuple[torch.Tensor, torch.Tensor], proposals: List[Instances]
):
    """
    Instead of using softmax, this version uses sigmoid as prediction 
    probabilty calculation to be valid for multi label tasks.
    """
    scores, _ = predictions
    num_inst_per_image = [len(p) for p in proposals]
    probs_sig = torch.sigmoid(scores)
    return probs_sig.split(num_inst_per_image, dim=0)

class CustomMapper(DatasetMapper):
    """
    Dataset mapper with changed call method to keep the annotations in the validation process.
    """

    def __call__(self, dataset_dict):
        """
        DatasetMapper.__call__ function is modified to control augmentations 
        for training and testing.
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.
        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)      
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image)  
        aug_input = T.AugInput(image, sem_seg=None)
        transforms = self.augmentations(aug_input)
        image = aug_input.image      
        image_shape = image.shape[:2]  # h, w
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))    
        # Keep the annotations for the validation purpose, otherwise it is directly inference/testing.
        # Pop only segmentation related part.
        if not self.is_train:
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict
                
        if "annotations" in dataset_dict:
            self._transform_annotations(dataset_dict, transforms, image_shape)

        return dataset_dict


class CustomEvaluator(DatasetEvaluator):
    """
    Custom evaluator to correctly handle multi labeled predictions. It is adapted from the YOLOv5
    (YOLOv4-CSP/YOLOR all have same evaluation approach) which is the PascalVOC evaluation. To have
    reasonable comparison between the YOLO and RCNN, the same evaluation approach is taken.
    """   

    def __init__(self, dataset_name):
        
        self._logger = logging.getLogger(__name__)
        self._cpu_device = torch.device("cpu")
        if torch.cuda.is_available():
            self._gpu = torch.device('cuda:0')
        else:
           self._gpu = self._cpu_device 
        self._metadata = MetadataCatalog.get(dataset_name)
        self.stats = []
        self.per_class = True
        self.dataset_name = dataset_name
        self.seen = 0
        self.output_list = np.array([])
    
    def reset(self):
        self._predictions = [] # class name -> list of prediction strings
   
    def ap_per_class(self, tp, conf, pred_cls, target_cls):
        """ 
        Compute the average precision, given the recall and precision curves.
        Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
        # Arguments
            tp:    True positives (nparray, nx1 or nx10).
            conf:  Objectness value from 0-1 (nparray).
            pred_cls: Predicted object classes (nparray).
            target_cls: True object classes (nparray).
        # Returns
            The average precision as computed in py-faster-rcnn.
        """
        # Sort by objectness
        i = np.argsort(-conf)
        tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

        # Find unique classes
        unique_classes = np.unique(target_cls)
        nc = unique_classes.shape[0] 
        px = np.linspace(0, 1, 1000)

        # Create Precision-Recall curve and compute AP for each class
        s = [unique_classes.shape[0], tp.shape[1]]  # number class, number iou thresholds 
        ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))

        for ci, c in enumerate(unique_classes):
            i = pred_cls == c
            n_gt = (target_cls == c).sum()  # Number of ground truth objects
            n_p = i.sum()  # Number of predicted objects

            if n_p == 0 or n_gt == 0:
                continue
            else:
                # Accumulate FPs and TPs
                fpc = (1 - tp[i]).cumsum(0)
                tpc = tp[i].cumsum(0)

                # Recall
                recall = tpc / (n_gt + 1e-16)  # recall curve
                r[ci] = np.interp(-px, -conf[i], recall[:, 0])  # negative x, xp because xp decreases

                # Precision
                precision = tpc / (tpc + fpc)  # precision curve
                p[ci] = np.interp(-px, -conf[i], precision[:, 0]) 

                # AP from recall-precision curve
                for j in range(tp.shape[1]):
                    ap[ci, j] = self.compute_ap(recall[:, j], precision[:, j])

                # Plot
                # fig, ax = plt.subplots(1, 1, figsize=(5, 5))
                # ax.plot(recall, precision)
                # ax.set_xlabel('Recall')
                # ax.set_ylabel('Precision')
                # ax.set_xlim(0, 1.01)
                # ax.set_ylim(0, 1.01)
                # fig.tight_layout()
                # fig.savefig('PR_curve.png', dpi=300)

        # Compute F1 score (harmonic mean of precision and recall)
        f1 = 2 * p * r / (p + r + 1e-16)

        return p, r, ap, f1, unique_classes.astype('int32')

    def compute_ap(self, recall, precision):
        """ 
        Compute the average precision, given the recall and precision curves.
        Code originally from https://github.com/rbgirshick/py-faster-rcnn.
        # Arguments
            recall:    The recall curve (list).
            precision: The precision curve (list).
        # Returns
            The average precision as computed in py-faster-rcnn.
        """
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([1.0], precision, [0.0]))
        mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        # i = np.where(mrec[1:] != mrec[:-1])[0]
        # # and sum (\Delta recall) * prec
        # ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate

        return ap       
        
    def process(self, inputs, outputs):
        """
        Processing the inputs/outputs and calculate stats as considering 2 identical bboxes per each
        label for ground-truths, and compare them with the predictions of the model which are also
        has the similar structure, i.e. giving out 2 quite close bboxes with 1 label information. 
        This also makes the calculation of the metrics more reasonable.
        """
        iouv = torch.linspace(0.5, 0.95, 10).to(self._gpu)                
        for input, output in zip(inputs, outputs):
            image_id = input["image_id"]
            gt_boxes = torch.tensor([box['bbox'] for box in input['annotations']], device=self._gpu)
            tcls = torch.tensor([box['category_id'] for box in input['annotations']], device=self._gpu).reshape(-1)
            instances = output["instances"]
            boxes = instances.pred_boxes.tensor
            self.scores = instances.scores
            classes = instances.pred_classes
            self.seen += 1
            nl = len(gt_boxes) * 2    # since one prediction has 2 bbox for each label.
            if len(instances) == 0:
                if nl:
                    self.stats.append((torch.zeros(0, 10, dtype=torch.bool), 
                                       torch.Tensor(), torch.Tensor(), tcls.tolist()))
                continue

            correct = torch.zeros(len(instances), 10, dtype=torch.bool, device=self._gpu)
            if nl:
                detected = []  # target indices
                tcls_tensor = tcls
                # target boxes
                tbox = gt_boxes.repeat_interleave(2, dim=0)   
                # Per target class
                for cls in torch.unique(tcls_tensor):
                    ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1) # target indices 
                    pi = (cls == classes).nonzero(as_tuple=False).view(-1)  # prediction indices

                    # Search for detections
                    if pi.shape[0]:
                        # Prediction to target ious
                        ious, i = box_iou(boxes[pi], tbox[ti]).max(1)  # best ious, indices

                        # Append detections
                        detected_set = set()
                        for j in (ious > iouv[0]).nonzero(as_tuple=False):
                            d = ti[i[j]]  # detected target
                            if d.item() not in detected_set:
                                detected_set.add(d.item())
                                detected.append(d)
                                correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                
                                if len(detected) == nl:  # all targets already located in image
                                    break
            self.stats.append((correct.cpu(), self.scores.cpu(), classes.cpu(), tcls.tolist()))
        # collecting the predictions
            self.output_list = np.concatenate([self.output_list, np.array([output, input['file_name']])])  
    
    def evaluate(self, img_ids=None):
        """
        Evaluates the processed inputs and outputs. 
        Returns the results and also prediction list. 
        """        
        names = MetadataCatalog.get(self.dataset_name).thing_classes
        cfg = get_cfg()
        nc = cfg.MODEL.ROI_HEADS.NUM_CLASSES        
        stats = [np.concatenate(x, 0) for x in zip(*self.stats)]
        p, r, f1, mp, mr, map50, map = 0., 0., 0., 0., 0., 0., 0.
        ap, ap_class = [], []
        
        if len(stats) and stats[0].any():
            p, r, ap, f1, ap_class = self.ap_per_class(*stats)
            p, r, ap50, ap = p[:, 0], r[:, 0], ap[:, 0], ap.mean(1)  # [P, R, AP@0.5, AP@0.5:0.95]
            mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
            nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
        else:
            nt = torch.zeros(1)

        # Print results
        s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
        pf = '%20s' + '%12.3g' * 6  # print format
        print(s)
        # print(s + '\n' + pf % ('all', self.seen, nt.sum(), mp, mr, map50, map))
        
        # Print results per class
        w_map50 = w_mP = w_mR = w_map = 0.0        
        if self.per_class and nc > 1 and len(stats):
            for i, c in enumerate(ap_class):
                w_map50 += nt[c] * ap50[i]
                w_mP += nt[c] * p[i]
                w_mR += nt[c] * r[i]
                w_map += nt[c] * ap[i]
                print(pf % (names[c], self.seen, nt[c], p[i], r[i], ap50[i], ap[i]))
            w_map50 = w_map50 / nt.sum()
            w_mP = w_mP / nt.sum()
            w_mR = w_mR / nt.sum()
            w_map = w_map / nt.sum()
            print(pf % ('Weighted all ', self.seen, nt.sum(), w_mP, w_mR, w_map50, w_map)) 

        maps = np.zeros(nc) + map
        for i, c in enumerate(ap_class):
            maps[c] = ap[i]
        
        ret = OrderedDict()
        ret["bbox"] = {"AP": map, "AP50": map50, "Precision": mp, "Recall": mr}        
        if self.dataset_name == 'test_set':
            return ret, self.output_list     
        else:
            return ret


def build_evaluator(cfg, dataset_name, output_folder=None):
    """
    To define CustomEvaluator for the default trainer/predictor.
    """    
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator_list = [CustomEvaluator(dataset_name)]

    return DatasetEvaluators(evaluator_list)


def read_image2(file_name, format=None):
    """
    Changed the original function to read the dataset in correct format according to dtld_parsing 
    repository (https://github.com/julimueller/dtld_parsing) from the author.
    """
    img = cv2.imread(file_name,-1)
    img = cv2.cvtColor(img, cv2.COLOR_BAYER_GB2RGB)
    #     Images are saved in 12 bit raw -> shift 4 bits
    img = np.right_shift(img, 4)
    img = img.astype(np.uint8)
    img = util._apply_exif_orientation(img)

    return img


def log2(pred_logits, gt_classes, prefix="fast_rcnn"):
    """
    Log the classification metrics to EventStorage. Changed 
    the original function for multi label task.
    """
    num_instances = gt_classes.numel()
    if num_instances == 0:
        return
    # taking top 2 prediction instead of 1  since there are 
    # 2 groups of classes: direction and bbulb number
    pred_classes = torch.topk(pred_logits, 2).indices 
    bg_class_ind = pred_logits.shape[1] - 1

    fg_inds = (gt_classes >= 0) & (gt_classes < bg_class_ind)
    fg_inds = torch.tensor([True if elem[0] == True and elem[1] == True \
                            else False for elem in fg_inds])
    num_fg = fg_inds.count_nonzero().numel()
    fg_gt_classes = gt_classes[fg_inds]
    fg_pred_classes = pred_classes[fg_inds]
    cntr = 0
    cntr += [1 for elem in (fg_pred_classes == bg_class_ind) \
                            if elem[0] == True and elem[1] == True].count(1)
    num_false_negative = int((fg_pred_classes == bg_class_ind).count_nonzero())
    num_false_negative -= cntr
    num_accurate = int((pred_classes.sort().values == gt_classes.sort().values).count_nonzero())
    fg_num_accurate = int((fg_pred_classes.sort().values == fg_gt_classes.sort().values).count_nonzero())

    storage = get_event_storage()
    storage.put_scalar(f"{prefix}/cls_accuracy", num_accurate / num_instances)
    if num_fg > 0:
        storage.put_scalar(f"{prefix}/fg_cls_accuracy", fg_num_accurate / num_fg)
        storage.put_scalar(f"{prefix}/false_negative", num_false_negative / num_fg)


def draw_labels(out=None, im_path=None, json_path=None, save_path=None, 
                dataset_dir='/multiverse/datasets/shared/DTLD'):
    """
    Displays the image in the actual size and draw labels according to inputs given.
    Additionally, save option is available.
    """    
    # dpi = matplotlib.rcParams['figure.dpi']   
    # height = 1024
    # width = 2048
        
    if json_path:
        with open(json_path) as file:
            parsed = json.load(file)
        images = parsed['images']            
        paths = [image['image_path'] for image in images]
        for idx, path in enumerate(paths):            
            path_split = path.split('/')
            path_split[0] = dataset_dir
            filename = '/'.join(path_split)
            img = read_image2(filename)
            labels = images[idx]['labels']            
            for label in labels:
                x = label['x']
                y = label['y']
                w = label['w']
                h = label['h']    
                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 
                              1, lineType=cv2.LINE_AA)
            if save_path:
                save_path = dataset_dir
                img_name = im_path.split('/')[-1][:-5]
                cv2.imwrite(save_path + img_name + '_labeled.tiff', 
                            cv2.cvtColor(img, cv2.COLOR_RGB2BGR))                            

                # continue

            # figsize = width / float(dpi), height / float(dpi)
            # fig = plt.figure(figsize=figsize)
            # ax = fig.add_axes([0, 0, 1, 1])
            # ax.axis('off')
            # ax.imshow(img)
            # plt.show()
            # _ = input('press smth')
    
    elif im_path:   
        txt_path = im_path[:-4] + 'txt'
        label_file = open(txt_path, 'r')
        labels = label_file.read().split('\n')
        labels.pop(-1)
        dw = 2048
        dh = 1024
        img = read_image2(im_path)
        for label in labels:
            xnorm = float(label.split(' ')[1])
            ynorm = float(label.split(' ')[2])
            wnorm = float(label.split(' ')[3])
            hnorm = float(label.split(' ')[4])
            xc = xnorm - wnorm/2
            yc = ynorm - hnorm/2
            x = int(xc*dw)
            y = int(yc*dh)
            w = int(wnorm*dw)
            h = int(hnorm*dh)
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        if out:
            labels = out['instances'].pred_boxes.tensor.tolist()
            classes = out['instances'].pred_classes.tolist()
            scores = out['instances'].scores.tolist()
            # img = read_image2(im_path)
            # save_path = '/multiverse/datasets/shared/DTLD/detectron2/bulb_seq1'
            os.makedirs(save_path, exist_ok=True)
            for label, cls, scr in zip(labels, classes, scores):
                x, y, x2, y2 = label[0], label[1], label[2], label[3]
                cv2.rectangle(img, (round(x), round(y)), (round(x2), round(y2)), (255, 255, 255), 2)
                dict_7 = {'1B': 0, 
                        '2B': 1,
                        '3B': 2,
                        'F': 3,
                        'B': 4,
                        'R': 5,
                        'L': 6}
                att = {v: k for k, v in dict_7.items()}
                label_text = '%s   %.1f' % (att[cls], scr)
                cv2.putText(img, label_text, (round(x2), round(y2) - 2), 
                            0, 2/3, [0, 255, 0], 
                            thickness=1, lineType=cv2.LINE_AA)         
            img_name = im_path.split('/')[-1][:-5]
            save2 = os.path.join(save_path, img_name + '_labeled.tiff')
            cv2.imwrite(save2, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        return
    
    elif out.size:
        labels = out[0]['instances'].pred_boxes.tensor.tolist()
        classes = out[0]['instances'].pred_classes.tolist()
        scores = out[0]['instances'].scores.tolist()
        # grouped = group_labels(out[0]['instances'])
        # import pdb;pdb.set_trace()
        img = read_image2(out[1])
        # save_path = '/multiverse/datasets/shared/DTLD/detectron2/seq12'
        os.makedirs(save_path, exist_ok=True)
        for label, cls, scr in zip(labels, classes, scores):
            x, y, x2, y2 = round(label[0]), round(label[1]), \
                           round(label[2]), round(label[3])
            cv2.rectangle(img, (x, y), (x2, y2), 
                          (0, 255, 0), 1, lineType=cv2.LINE_AA)
            dict_7 = {'1': 0, 
                      '2': 1,
                      '3': 2,
                      'f': 3,
                      'b': 4,
                      'r': 5,
                      'L': 6}
            att = {v: k for k, v in dict_7.items()}
            # label_text = '%s %.1f' % (att[cls], scr)
            label_text = att[cls]
            scale = (label[2] - label[0])
            try: 
                int(label_text)  
                label_x = x + 4
            except:
                label_x = x - 2
            
            cv2.putText(img, label_text, (label_x, y-2), 
                        0, scale/12, [255, 255, 255], 
                        thickness=1, lineType=cv2.LINE_AA) 

        # txt_path = out[1][:-4] + 'txt'
        # if os.path.exists(txt_path):
        #     label_file = open(txt_path, 'r')
        #     labels = label_file.read().split('\n')
        #     labels.pop(-1)
        #     dw = 2048
        #     dh = 1024
        #     for label in labels:
        #         xnorm = float(label.split(' ')[1])
        #         ynorm = float(label.split(' ')[2])
        #         wnorm = float(label.split(' ')[3])
        #         hnorm = float(label.split(' ')[4])
        #         xc = xnorm - wnorm/2
        #         yc = ynorm - hnorm/2
        #         x = int(xc*dw)
        #         y = int(yc*dh)
        #         w = int(wnorm*dw)
        #         h = int(hnorm*dh)
        #         cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 
        #                     1, lineType=cv2.LINE_AA)
        img_name = out[1].split('/')[-1][:-5]
        save2 = os.path.join(save_path, img_name + '_labeled.tiff')
        cv2.imwrite(save2, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        return

    # figsize = width / float(dpi), height / float(dpi)
    # fig = plt.figure(figsize=figsize)
    # ax = fig.add_axes([0, 0, 1, 1])
    # ax.axis('off')
    # ax.imshow(img)
    # plt.show()


def get_wh(X_train, opt):
    """
    Calculate the widths and heights of the labels scaled 
    to the image size used during training. The output
    is used by the :kmeans(): function.
    """    
    obj_count = 0 
    wh = []    
    for image in X_train:   
        height, width = 1024, 2048    
             
        for anno in image['labels']:
            x = anno['x']
            y = anno['y']
            w = anno['w']
            h = anno['h']
            if w == 0 or h == 0:
                continue
            # Not allowing the bbox coordinates outside of the image 
            if x < 0:
                w = w + x
                x = 0                       
            elif y < 0:
                h = h + y
                y = 0
            elif x + w > 2048:
                w = 2048 - x
            elif y + h > 1024:
                h = 1024 - y                 
                
            obj_count += 1           
            wh.append([w*opt.img_size[1]/width, h*opt.img_size[0]/height])
    
    print(obj_count)
    return wh
    

def iou(box, clusters):
    """
    Calculates the Intersection over Union (IoU) between a box and k clusters.
    :param box: tuple or array, shifted to the origin (i. e. width and height)
    :param clusters: numpy array of shape (k, 2) where k is the number of clusters
    :return: numpy array of shape (k, 0) where k is the number of clusters
    """
    x = np.minimum(clusters[:, 0], box[0])
    y = np.minimum(clusters[:, 1], box[1])
    if np.count_nonzero(x == 0) > 0 or np.count_nonzero(y == 0) > 0:
        raise ValueError("Box has no area")

    intersection = x * y
    box_area = box[0] * box[1]
    cluster_area = clusters[:, 0] * clusters[:, 1]
    iou_ = intersection / (box_area + cluster_area - intersection)

    return iou_


def avg_iou(boxes, clusters):
    """
    Calculates the average Intersection over Union (IoU) between a numpy array 
    of boxes and k clusters.
    :param boxes: numpy array of shape (r, 2), where r is the number of rows
    :param clusters: numpy array of shape (k, 2) where k is the number of clusters
    :return: average IoU as a single float
    """
    return np.mean([np.max(iou(boxes[i], clusters)) for i in range(boxes.shape[0])])


def translate_boxes(boxes):
    """
    Translates all the boxes to the origin.
    :param boxes: numpy array of shape (r, 4)
    :return: numpy array of shape (r, 2)
    """
    new_boxes = boxes.copy()
    for row in range(new_boxes.shape[0]):
        new_boxes[row][2] = np.abs(new_boxes[row][2] - new_boxes[row][0])
        new_boxes[row][3] = np.abs(new_boxes[row][3] - new_boxes[row][1])
    return np.delete(new_boxes, [0, 1], axis=1) 
    

def kmeans(boxes, k, dist=np.median):
    """
    Calculates k-means clustering with the Intersection over Union (IoU) metric.
    :param boxes: numpy array of shape (r, 2), where r is the number of rows
    :param k: number of clusters
    :param dist: distance function
    :return: numpy array of shape (k, 2)
    """
    rows = boxes.shape[0]
    distances = np.empty((rows, k))
    last_clusters = np.zeros((rows,))
    np.random.seed()
    # the Forgy method will fail if the whole array contains the same rows
    clusters = boxes[np.random.choice(rows, k, replace=False)]

    while True:
        for row in range(rows):
            distances[row] = 1 - iou(boxes[row], clusters)

        nearest_clusters = np.argmin(distances, axis=1)

        if (last_clusters == nearest_clusters).all():
            break

        for cluster in range(k):
            clusters[cluster] = dist(boxes[nearest_clusters == cluster], axis=0)

        last_clusters = nearest_clusters

    return clusters


# def group_labels(instances):
#     """
#     Take instances as input from detectron2 predictor, 
#     group them as there are 2 classes for each objects. 
#     The model returns 1 class * 2 bbox. This function 
#     basically convert this to 2 class * 1 bbox and 
#     returns grouped list of predictions.
#     """      
#     boxes = instances.pred_boxes.tensor
#     _, indices = boxes.T[0].sort()
#     boxes = instances[indices].pred_boxes.tensor
#     pred_classes = instances[indices].pred_classes
#     scores = instances[indices].scores

#     labels = [np.concatenate([box, [cls], [scr]])
#               for box, cls, scr          
#               in zip(boxes.tolist(), 
#                      pred_classes.tolist(), 
#                      scores.tolist())]    

#     grouped_list = np.split(labels, 
#                             np.where((np.abs(np.diff(boxes.T[0].cpu())) > 1.5))[0] + 1)

#     return grouped_list

import scipy
from scipy.cluster.hierarchy import fclusterdata
def group_labels(instances):
    """
    Take instances as input from detectron2 predictor, 
    group them as there are 2 classes for each objects. 
    The model returns 1 class * 2 bbox. This function 
    basically convert this to 2 class * 1 bbox and 
    returns grouped list of predictions.
    """      
    boxes = instances.pred_boxes.tensor
    _, indices = boxes.T[0].sort()
    boxes = instances[indices].pred_boxes.tensor
    pred_classes = instances[indices].pred_classes
    scores = instances[indices].scores

    labels = np.array(
        [np.concatenate([box, [cls], [scr]])
        for box, cls, scr          
        in zip(boxes.tolist(), 
               pred_classes.tolist(),             
               scores.tolist())]
    ) 
    mean_width = (boxes[:, 2] - boxes[:, 0]).mean()
    thres = 3**2 if mean_width < 10 else 5**2
    cluster_ind = fclusterdata(boxes[:, :4].cpu(), thres, criterion='distance', metric='euclidean') 

    # import pdb;pdb.set_trace()

    grouped_list = [labels[cluster_ind==idx+1, :] 
                    for idx in range(max(cluster_ind))]
    # grouped_list = np.split(labels, 
    #                         np.where((np.abs(np.diff(boxes.T[0].cpu())) > 1.5))[0] + 1)

    return grouped_list


def get_att(label_group):
    """
    Set the aspect and direction for the label groups, for all possible 
    prediction outcomes. 
    Returns bbox, aspect and direction for the given label group.
    """
    dict_7 = {'one_aspect': 0, 
              'two_aspects': 1,
              'three_aspects': 2,
              'front': 3,
              'back': 4,
              'right': 5,
              'left': 6}
    att = {v: k for k, v in dict_7.items()}
  
    # if len(label_group) == 1:
    #     # bbox = label_group[0][:4] # xyxy
    #     # cls = label_group[0][4]                        
    #     # if cls < 3:
    #     #     aspect = att[cls]
    #     #     direction = att[3]
    #     # else:
    #     #     aspect = att[2]
    #     #     direction = att[cls]
    #     print('hello')
    #     return

    # else:
    
    bbox = label_group[0][:4] # xyxy
    classes = np.array([int(anno[4]) for anno in label_group])
    scores = np.array([anno[5] for anno in label_group])
    dir_ind = np.where(classes >= 3)[0]
    asp_ind =  np.where(classes < 3)[0]
    dir_scores = scores[dir_ind]
    dir_scr_idx = np.argsort(dir_scores)
    dir_max = dir_ind[dir_scr_idx][-1] if dir_ind.size else np.array([])
    asp_scores = scores[asp_ind]
    asp_scr_idx = np.argsort(asp_scores)
    asp_max = asp_ind[asp_scr_idx][-1] if asp_ind.size else np.array([])
    direction = att[classes[dir_max].item()] if dir_max.size else att[3]
    aspect = att[classes[asp_max].item()] if asp_max.size else att[2]

    return bbox, aspect, direction


def save_pred(input_json, outputs, save_path=None):
    """
    Used for saving the results of the TEST SET. For detection, use 
    save_detect().
    Given the input json file and the prediction outputs, generates json file 
    as in the DTLD format (like input file).
    Returns the dictionary or saves it to the json file to indicated save path. 
    """
    with open(input_json) as file:
        parsed = json.load(file)
    images = parsed['images']

    for image, output in zip(images, outputs): 
        instances = output[0]['instances']  
        grouped_list = group_labels(instances)
        label_dict_list = []    
        for group in grouped_list:
            if group.size == 0 or len(group) == 1:
                continue

            bbox, aspect, direction = get_att(group)
            attributes = {"aspects": aspect,  
                          "direction": direction,
                          "occlusion": '', 
                          "orientation": '', 
                          "reflection": '',
                          "pictogram": '',
                          "relevance": '',
                          "state": ''}

            x, y, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]            
            w, h = x2 - x, y2 - y
            label_dict = {"attributes": attributes,
                          "h": int(h),
                          "w": int(w),
                          "x": int(x),
                          "y": int(y),
                          "track_id": '',
                          "unique_id": ''}                          
            label_dict_list.append(label_dict)

        image["labels"] = label_dict_list

    if save_path:
        import time
        timestr = time.strftime("%Y_%m_%d_%H%M") + '_'
        input_name = input_json.split('/')[-1][:-5] + '_'
        path = os.path.join(save_path, timestr + input_name + 'inference_results_best.json')
        with open(path, 'w+') as f:
            json.dump(parsed, f, indent=4) 
        print('Saved to', path)       
        return None
        
    else:
        return parsed


def save_detect(input_json, outputs, save_path=None):
    """
    Given the input json file and the detection outputs, generates json file 
    as in the DTLD format (like defaul_inp.json file).
    Returns the dictionary or saves it to the json file to indicated save path. 
    """
    with open(input_json) as file:
        parsed = json.load(file)
    images = parsed['images']
    del_list = []
    for idx, (image, output) in enumerate(zip(images, outputs)): 
        # if image['image_path'] == './Berlin/Berlin1/2015-04-17_10-58-12/DE_BBBR667_2015-04-17_10-58-23-369480_k0.tiff':
        #     import pdb;pdb.set_trace()
        instances = output['instances']
        
        if len(instances):
            grouped_list = group_labels(instances)

        else:
            # # to distinguish images with no detection
            # attributes = {"aspects": '',  
            #               "direction": '',
            #               "occlusion": '', 
            #               "orientation": '', 
            #               "reflection": '',
            #               "pictogram": '',
            #               "relevance": '',
            #               "state": ''}

            # label_dict = {"attributes": attributes,
            #               "h": 0,
            #               "w": 0,
            #               "x": 0,
            #               "y": 0,
            #               "track_id": '',
            #               "unique_id": ''}                          
            # image["labels"] = [label_dict]
            # import pdb;pdb.set_trace()
            del_list.append(idx)
            continue

        label_dict_list = []    
        for group in grouped_list:
            if len(group) == 1:
                continue

            bbox, aspect, direction = get_att(group)
            attributes = {"aspects": aspect,  
                          "direction": direction,
                          "occlusion": '', 
                          "orientation": '', 
                          "reflection": '',
                          "pictogram": '',
                          "relevance": '',
                          "state": ''}

            x, y, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]            
            w, h = x2 - x, y2 - y
            label_dict = {"attributes": attributes,
                          "h": int(h),
                          "w": int(w),
                          "x": int(x),
                          "y": int(y),
                          "track_id": '',
                          "unique_id": ''}                          
            label_dict_list.append(label_dict)
        
        if label_dict_list:
            image["labels"] = label_dict_list
        else:
            del_list.append(idx)
        
    images = np.delete(images, del_list).tolist()
    parsed['images'] = images
    if save_path:
        import time
        timestr = time.strftime("%Y_%m_%d_%H%M") + '_'
        input_name = input_json.split('/')[-1][:-5] + '_'
        path = os.path.join(save_path, timestr + input_name + 'detection.json')
        with open(path, 'w+') as f:
            json.dump(parsed, f, indent=4) 
        print('Saved to', path)       
        return path
        
    else:
        return parsed    



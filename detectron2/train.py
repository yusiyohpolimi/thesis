import numpy as np
import os
import logging
import torch 
import argparse

import detectron2.modeling.roi_heads.fast_rcnn as fast
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.layers import cat
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.engine.train_loop import HookBase
from detectron2.data.dataset_mapper import *
from detectron2.data import build_detection_train_loader, build_detection_test_loader
from detectron2.modeling import build_model

from fvcore.nn import giou_loss, smooth_l1_loss
from torch.nn import functional as F
from general import *
from detectron2.engine.hooks import PeriodicCheckpointer


def box_reg_loss2(self, proposal_boxes, gt_boxes, pred_deltas, gt_classes):
    """
    Changed the original function to have multi-label gt_classes and determine foreground indices
    by using them.
    """
    box_dim = proposal_boxes.shape[1]  # 4 or 5
    # Regression loss is only computed for foreground proposals (those matched to a GT)
    fg_inds = (gt_classes >= 0) & (gt_classes < self.num_classes)
    fg_inds = torch.tensor([True if elem[0] == True and elem[1] == True \
                                         else False for elem in fg_inds])   
    
    if pred_deltas.shape[1] == box_dim:  # cls-agnostic regression
        fg_pred_deltas = pred_deltas[fg_inds]
    else:
        # 2 classes but they should have same bbox thus take 1 gt.
        fg_pred_deltas = pred_deltas.view(-1, self.num_classes, box_dim)[fg_inds, gt_classes[fg_inds, 0]]

    if self.box_reg_loss_type == "smooth_l1":
        gt_pred_deltas = self.box2box_transform.get_deltas(
            proposal_boxes[fg_inds],
            gt_boxes[fg_inds],
        )
        loss_box_reg = smooth_l1_loss(
            fg_pred_deltas, gt_pred_deltas, self.smooth_l1_beta, reduction="sum"
        )
    elif self.box_reg_loss_type == "giou":
        fg_pred_boxes = self.box2box_transform.apply_deltas(
            fg_pred_deltas, proposal_boxes[fg_inds]
        )
        loss_box_reg = giou_loss(fg_pred_boxes, gt_boxes[fg_inds], reduction="sum")
    else:
        raise ValueError(f"Invalid bbox reg loss type '{self.box_reg_loss_type}'")
    # The reg loss is normalized using the total number of regions (R), not the number
    # of foreground regions even though the box regression loss is only defined on
    # foreground regions. Why? Because doing so gives equal training influence to
    # each foreground example. To see how, consider two different minibatches:
    #  (1) Contains a single foreground region
    #  (2) Contains 100 foreground regions
    # If we normalize by the number of foreground regions, the single example in
    # minibatch (1) will be given 100 times as much influence as each foreground
    # example in minibatch (2). Normalizing by the total number of regions, R,
    # means that the single example in minibatch (1) and each of the 100 examples
    # in minibatch (2) are given equal influence.
    return loss_box_reg / max(gt_classes.shape[0], 1.0)  # return 0 if empty
    

def losses2(self, predictions, proposals):
    """
    Changed the original function to use binary cross entropy loss with logits as the use case is 
    multi label task. Instead of one-hot encoded ground truths, multi-hot encoded list implemented. 
    """
    scores, proposal_deltas = predictions

    # parse classification outputs    
    gt_classes = (
        cat([p.gt_classes for p in proposals], dim=0) if len(proposals) else torch.empty(0)
    )
    log2(scores, gt_classes) # using the new log function defined

    # parse box regression outputs
    if len(proposals):
        proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim=0)  # Nx4
        assert not proposal_boxes.requires_grad, "Proposals should not require gradients!"
        # If "gt_boxes" does not exist, the proposals must be all negative and
        # should not be included in regression loss computation.
        # Here we just use proposal_boxes as an arbitrary placeholder because its
        # value won't be used in self.box_reg_loss().
        gt_boxes = cat(
            [(p.gt_boxes if p.has("gt_boxes") else p.proposal_boxes).tensor for p in proposals],
            dim=0,
        )
    else:
        proposal_boxes = gt_boxes = torch.empty((0, 4), device=proposal_deltas.device)    

    #Define ground-truth multihot-encoded vector for BCELoss    
    gt_hot = torch.zeros(gt_classes.shape[0], scores.shape[1], device=proposal_deltas.device)
    for idx in range(gt_classes.shape[0]):
        gt_hot[idx][gt_classes[idx]] = 1    
    
    #Use BCELoss with logits (BCELoss + Sigmoid basically) for multi-label tasks
    losses = {
        "loss_cls": F.binary_cross_entropy_with_logits(scores, gt_hot, reduction="mean"),
        "loss_box_reg": self.box_reg_loss(
            proposal_boxes, gt_boxes, proposal_deltas, gt_classes
        ),
    }
    return {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}


class BestCheckpointer(HookBase):
    """
    By default, detectron2 does not save the best model but last. Implemented this hook to have
    best model saved.
    """

    def before_train(self):
        self.best_metric = 0.0
        self.logger = logging.getLogger("detectron2.trainer")
        self.logger.info("######## Running best check pointer")

    def after_step(self):
        metric_name = 'bbox/AP50'
        if metric_name in self.trainer.storage._history:
            eval_metric, batches = self.trainer.storage.history(metric_name)._data[-1]
            if self.best_metric < eval_metric:
                self.best_metric = eval_metric
                self.logger.info(f"######## New best metric: {self.best_metric}")
                self.trainer.checkpointer.save(f"model_best_ap50")


class Trainer(DefaultTrainer):
    """
    Defining a customized trainer class with the same properties of DefaulTrainer and also with
    custom evaluator and custom hooks.
    """
    
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return build_evaluator(cfg, dataset_name, output_folder)

    @classmethod
    def build_train_loader(cls, cfg): 
        return build_detection_train_loader(cfg, 
                                            mapper=CustomMapper(cfg, is_train=True, 
                                                                augmentations=[T.Resize((opt.img_size[0], 
                                                                                         opt.img_size[1])),
                                                                               T.RandomFlip()]))

    @classmethod
    def build_model(cls, cfg):
        model = build_model(cfg)
        return model

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, 
                                           dataset_name, 
                                           mapper=CustomMapper(cfg, is_train=False, 
                                                               augmentations=[T.Resize((opt.img_size[0], 
                                                                                        opt.img_size[1]))]))                                                                                         
       
    def build_hooks(self):                  
        ret = super().build_hooks()     # adding the new hook to default ones
        ret[-3] = PeriodicCheckpointer(self.checkpointer, 500, max_to_keep=3)
        ret.append(BestCheckpointer())
        return ret


def main(opt): 
    utils.annotations_to_instances = anno_to_instances 
    utils.read_image = read_image2
    utils.transform_instance_annotations = transform_instance_annotations2
    fast.FastRCNNOutputLayers.box_reg_loss = box_reg_loss2
    fast.FastRCNNOutputLayers.losses = losses2
    fast.FastRCNNOutputLayers.predict_probs = predict_probs2
    fast._log_classification_stats = log2

    task = 'train'
    json_path = opt.data
    X_train, X_val = DTLD_splitter(json_path, task)
    data_size = [len(X_train), len(X_val)]
    train_data = round(data_size[0] * opt.ratio)
    val_data = round(data_size[1] * opt.ratio)

    DatasetCatalog.register("train_set", lambda d=X_train[:train_data]: get_DTLD(d))
    DatasetCatalog.register("val_set", lambda d=X_val[:val_data]: get_DTLD(d))  
    MetadataCatalog.get("train_set").set(thing_classes=['1B', '2B', '3B', 'F', 'B', 'R', 'L'])
    MetadataCatalog.get("val_set").set(thing_classes=['1B', '2B', '3B', 'F', 'B', 'R', 'L'])

    cfg = get_cfg()
    if opt.model == 'R50':
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    elif opt.model == 'X101':
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))

    cfg.DATASETS.TRAIN = ("train_set",)
    cfg.DATASETS.TEST = ("val_set",)
    cfg.DATALOADER.NUM_WORKERS = 8
    if opt.weights == 'zoo' and not opt.resume:
        if opt.model == 'R50':
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
        elif opt.model == 'X101':
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")  
    else:
        cfg.MODEL.WEIGHTS = opt.weights

    cfg.SOLVER.IMS_PER_BATCH = opt.batch_size   
    cfg.SOLVER.BASE_LR = 0.01   
    NUM_GPUS = 1
    single_iteration = NUM_GPUS * cfg.SOLVER.IMS_PER_BATCH
    train_len = len(DatasetCatalog.get('train_set'))
    iterations_for_one_epoch = np.ceil(train_len / single_iteration)
    tot_epoch = opt.epochs

    cfg.SOLVER.MAX_ITER = int(iterations_for_one_epoch * tot_epoch)
    cfg.SOLVER.WARMUP_ITERS = int(iterations_for_one_epoch) # should put this otherwise it is 1k 3*5*8
    cfg.SOLVER.STEPS = []        # do not decay learning rate
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 7 
    det2_path = 'det2/'
    cfg.OUTPUT_DIR = det2_path + opt.logdir    
    cfg.TEST.EVAL_PERIOD = int(iterations_for_one_epoch)
    cfg.SOLVER.CHECKPOINT_PERIOD = 500
    cfg.SOLVER.LR_SCHEDULER_NAME = 'WarmupCosineLR'     # cosine annealing lr scheduler
    
    if opt.anchors:
        cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[8], [12], [21], [32], [64], [128], [256], [512]]
        cfg.MODEL.RPN.IN_FEATURES = ['p2', 'p2', 'p2', 'p2', 'p3', 'p4', 'p5', 'p6']
        cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[3.25]]  
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = Trainer(cfg) 
    trainer.resume_or_load(resume=opt.resume)
    setup_logger(output=cfg.OUTPUT_DIR)
    logger = logging.getLogger('detectron2.cfg')
    logger.info(cfg)
    # Saving the cfg to the same folder with weights to use them later in the inference.
    cfg2save = cfg.dump()
    save_path = os.path.join(cfg.OUTPUT_DIR, 'config.yaml')
    with open(save_path, 'w') as f:
        f.write(cfg2save)

    trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='zoo', help='initial weights path')
    parser.add_argument('--model', type=str, default='R50', help='R50, X101')
    parser.add_argument('--data', type=str, 
                        default='/multiverse/datasets/shared/DTLD/v2.0/v2.0/DTLD_train.json', 
                        help='json file of trainset')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='train,test sizes')
    parser.add_argument('--logdir', type=str, default='', help='logging directory')
    parser.add_argument('--ratio', type=float, default=1.0, help='ratio of dataset to train/val')
    parser.add_argument('--resume', action='store_true', help='to resume training from a checkpoint')
    parser.add_argument('--anchors', action='store_true', help='use custom anchors')
    opt = parser.parse_args()

    main(opt)





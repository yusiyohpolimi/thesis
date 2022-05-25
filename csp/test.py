import argparse
import glob
import json
import os
import shutil
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

from models.experimental import attempt_load
from utils.datasets import create_dataloader
from utils.general import (
    coco80_to_coco91_class, check_file, check_img_size, compute_loss, non_max_suppression,
    scale_coords, xyxy2xywh, clip_coords, plot_images, xywh2xyxy, box_iou, output_to_target, ap_per_class)
from utils.torch_utils import select_device, time_synchronized


def test(data,
         weights=None,
         batch_size=16,
         imgsz=640,
         conf_thres=0.001,
         iou_thres=0.6,  # for NMS
         save_json=False,
         single_cls=False,
         augment=False,
         verbose=False,
         model=None,
         dataloader=None,
         save_dir='',
         merge=False,
         save_txt=False):
    # Initialize/load model and set device
    training = model is not None
    if training:  # called by train.py
        device = next(model.parameters()).device  # get model device

    else:  # called directly
        device = select_device(opt.device, batch_size=batch_size)
        merge, save_txt = opt.merge, opt.save_txt  # use Merge NMS, save *.txt labels
        if save_txt:
            out = Path('inference/output')
            if os.path.exists(out):
                shutil.rmtree(out)  # delete output folder
            os.makedirs(out)  # make new output folder

        # Remove previous
        for f in glob.glob(str(Path(save_dir) / 'test_batch*.jpg')):
            os.remove(f)

        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size

    # Half
    half = device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()

    # Configure
    model.eval()
    with open(data) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)  # model dict
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # Dataloader
    if not training:
        img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
        _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
        path = data['test'] if opt.task == 'test' else data['val']  # path to val/test images
        opt.shuffle = False
        dataloader = create_dataloader(path, imgsz, batch_size, model.stride.max(), opt,
                                       hyp=None, augment=False, cache=False, pad=0.5, rect=True)[0]    ####################################################

    seen = 0
    names = model.names if hasattr(model, 'names') else model.module.names
    coco91class = coco80_to_coco91_class()
    s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    p, r, f1, mp, mr, map50, map, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class = [], [], [], []
    output_list = []
    #model = model.to(memory_format=torch.channels_last)
    for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
        img = img.to(device, non_blocking=True)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        nb, _, height, width = img.shape  # batch size, channels, height, width
        whwh = torch.Tensor([width, height, width, height]).to(device)

        # Disable gradients
        with torch.no_grad():
            # Run model
            t = time_synchronized()
            inf_out, train_out = model(img, augment=augment)  # inference and training outputs
            #inf_out, train_out = model(img.to(memory_format=torch.channels_last), augment=augment)  # inference and training outputs
            t0 += time_synchronized() - t
            # import pdb; pdb.set_trace()
            # Compute loss
            if training:  # if model has loss hyperparameters
                loss += compute_loss([x.float() for x in train_out], targets, model)[1][:3]  # GIoU, obj, cls

            # Run NMS
            t = time_synchronized()
            output = non_max_suppression(inf_out, conf_thres=conf_thres, iou_thres=iou_thres, merge=merge, agnostic=True)
            t1 += time_synchronized() - t
            output_list.append(output)
            
            # Save bboxes to generate json in DTLD format

        # Statistics per image
        for si, pred in enumerate(output):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            # pdb.set_trace()
            tcls = labels[:, 0].tolist() if nl else []  # target class
            seen += 1

            if pred is None:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Append to text file
            if save_txt:
                gn = torch.tensor(shapes[si][0])[[1, 0, 1, 0]]  # normalization gain whwh
                txt_path = str(out / Path(paths[si]).stem)
                pred[:, :4] = scale_coords(img[si].shape[1:], pred[:, :4], shapes[si][0], shapes[si][1])  # to original
                for *xyxy, conf, cls in pred:
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    with open(txt_path + '.txt', 'a') as f:
                        f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

            # Clip boxes to image bounds
            clip_coords(pred, (height, width))
            
            # Append to pycocotools JSON dictionary
            if save_json:
                # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
                image_id = Path(paths[si]).stem
                box = pred[:, :4].clone()  # xyxy
                scale_coords(img[si].shape[1:], box, shapes[si][0], shapes[si][1])  # to original shape
                box = xyxy2xywh(box)  # xywh
                box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
                for p, b in zip(pred.tolist(), box.tolist()):
                    jdict.append({'image_id': int(image_id) if image_id.isnumeric() else image_id,
                                  'category_id': coco91class[int(p[5])],
                                  'bbox': [round(x, 3) for x in b],
                                  'score': round(p[4], 5)})

            # Assign all predictions as incorrect
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
            if nl:
                detected = []  # target indices
                tcls_tensor = labels[:, 0]
                # target boxes
                tbox = xywh2xyxy(labels[:, 1:5]) * whwh
                # Per target class
                for cls in torch.unique(tcls_tensor):
                    ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # prediction indices
                    pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # target indices

                    # Search for detections
                    if pi.shape[0]:
                        # Prediction to target ious
                        ious, i = box_iou(pred[pi, :4], tbox[ti]).max(1)  # best ious, indices

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
            # Append statistics (correct, conf, pcls, tcls)
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

        # Plot images
        # if batch_i < 1:
        #     f = Path(save_dir) / ('test_batch%g_gt.jpg' % batch_i)  # filename
        #     plot_images(img, targets, paths, str(f), names)  # ground truth
        #     f = Path(save_dir) / ('test_batch%g_pred.jpg' % batch_i)
        #     plot_images(img, output_to_target(output, width, height), paths, str(f), names)  # predictions
    
    # import pdb; pdb.set_trace()
    
    if opt.dtld_json:
        import json

        input_json = opt.dtld_json[0]
        with open(input_json) as file:
            parsed = json.load(file)
        image_list = np.split(parsed['images'], 
                              np.arange(opt.batch_size, len(parsed['images']), 
                              opt.batch_size))
        dict_7 = {'one_aspect': 0, 
                'two_aspects': 1,
                'three_aspects': 2,
                'front': 3,
                'back': 4,
                'right': 5,
                'left': 6}
        att = {v: k for k, v in dict_7.items()}                            
        parsed_idx = 0
        for images, outputs in zip(image_list, output_list):  # batched inputs/results
            for si, (image, output) in enumerate(zip(images, outputs)): 
                if output is None:   # to distunguish images with no predictions  
                    # attributes = {"aspects": '' ,
                    #             "direction": '',
                    #             "occlusion": '', 
                    #             "orientation": '', 
                    #             "reflection": '',
                    #             "pictogram": '',
                    #             "relevance": '',
                    #             "state": ''}     
                    # label_dict = {"attributes": attributes,                                
                    #             "h": 0,
                    #             "w": 0,
                    #             "x": 0,
                    #             "y": 0,
                    #             "track_id": '',
                    #             "unique_id": ''}
                    # image["labels"] = [label_dict]
                    # parsed['images'][parsed_idx] = image
                    # parsed_idx += 1
                    continue
                import pdb;pdb.set_trace()         
                output[:, :4] = scale_coords(img[0].shape[1:], output[:, :4], shapes[0][0], shapes[0][1])
                _, indices =  output.T[0].sort()
                output =  output[indices]
                output = np.column_stack((output[:, :4].cpu(), output[:, 4:].cpu()))
                grouped_list = np.split(output, 
                                        np.where(
                                                (np.abs(np.diff(output.T[0])) > 1.5) 
                                                * (np.abs(np.diff(output.T[1])) > 1.5)
                                                )[0] + 1
                                        )
                
                label_dict_list = []       
                for label_group in grouped_list:
                    if len(label_group) == 1:
                        continue
                    
                    else:
                        bbox = label_group[0][:4] # xyxy
                        classes = np.array([int(anno[5]) for anno in label_group])
                        scores = np.array([anno[4] for anno in label_group])
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

                    attributes = {"aspects": aspect ,
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
                parsed['images'][parsed_idx] = image
                parsed_idx += 1
        
        path_split = weights[0].split('/')
        save_path = '/'.join(path_split[:-1])
        f = path_split[-1].replace('.pt', '.json')
        save_path = os.path.join(save_path, f)
        with open(save_path, 'w+') as file:
            json.dump(parsed, file, indent=4)
    
    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats)
        p, r, ap50, ap = p[:, 0], r[:, 0], ap[:, 0], ap.mean(1)  # [P, R, AP@0.5, AP@0.5:0.95]
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    pf = '%20s' + '%12.3g' * 6  # print format
    # Print results per class
    w_map50 = w_mP = w_mR = w_map = 0.0         
    if verbose and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            w_map50 += nt[c] * ap50[i]
            w_mP += nt[c] * p[i]
            w_mR += nt[c] * r[i]
            w_map += nt[c] * ap[i]
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))
        w_map50 = w_map50 / nt.sum()
        w_mP = w_mP / nt.sum()
        w_mR = w_mR / nt.sum()
        w_map = w_map / nt.sum()
        print(pf % ('weighted_all', seen, nt.sum(), w_mP, w_mR, w_map50, w_map)) 

    # Print speeds
    t = tuple(x / seen * 1E3 for x in (t0, t1, t0 + t1)) + (imgsz, imgsz, batch_size)  # tuple
    if not training:
        print('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % t)

    # Save JSON
    if save_json and len(jdict):
        f = 'detections_val2017_%s_results.json' % \
            (weights.split(os.sep)[-1].replace('.pt', '') if isinstance(weights, str) else '')  # filename
        print('\nCOCO mAP with pycocotools... saving %s...' % f)
        with open(f, 'w') as file:
            json.dump(jdict, file)

        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            imgIds = [int(Path(x).stem) for x in dataloader.dataset.img_files]
            cocoGt = COCO(glob.glob('../coco/annotations/instances_val*.json')[0])  # initialize COCO ground truth api
            cocoDt = cocoGt.loadRes(f)  # initialize COCO pred api
            cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
            cocoEval.params.imgIds = imgIds  # image IDs to evaluate
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
            map, map50 = cocoEval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
        except Exception as e:
            print('ERROR: pycocotools unable to run: %s' % e)

    # Return results
    model.float()  # for training
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--weights', nargs='+', type=str, default='yolov4-p5.pt', help='model.pt path(s)')
    parser.add_argument('--data', type=str, default='data/coco128.yaml', help='*.data path')
    parser.add_argument('--batch-size', type=int, default=32, help='size of each image batch')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.65, help='IOU threshold for NMS')
    parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--task', default='val', help="'val', 'test', 'study'")
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--merge', action='store_true', help='use Merge NMS')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--dtld-json', nargs='+', type=str, default='', 
                        help='save results in given DTLD json format')

    opt = parser.parse_args()
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.data = check_file(opt.data)  # check file
    print(opt)

    if opt.task in ['val', 'test']:  # run normally
        test(opt.data,
             opt.weights,
             opt.batch_size,
             opt.img_size,
             opt.conf_thres,
             opt.iou_thres,
             opt.save_json,
             opt.single_cls,
             opt.augment,
             opt.verbose)

    elif opt.task == 'study':  # run over a range of settings and save/plot
        for weights in ['']:
            f = 'study_%s_%s.txt' % (Path(opt.data).stem, Path(weights).stem)  # filename to save to
            x = list(range(352, 832, 64))  # x axis
            y = []  # y axis
            for i in x:  # img-size
                print('\nRunning %s point %s...' % (f, i))
                r, _, t = test(opt.data, weights, opt.batch_size, i, opt.conf_thres, opt.iou_thres, opt.save_json)
                y.append(r + t)  # results and times
            np.savetxt(f, y, fmt='%10.4g')  # save
        os.system('zip -r study.zip study_*.txt')
        # plot_study_txt(f, x)  # plot

import argparse
import os
import platform
import shutil
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, plot_one_box, strip_optimizer)
from utils.torch_utils import select_device, load_classifier, time_synchronized


def detect(save_img=False):
    out, source, weights, view_img, save_txt, imgsz = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    device = select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    out = os.path.join('csp', out)
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    pred_list = []
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()
        pred_list.append(pred[0])

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                for *xyxy, conf, cls in det:
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

                    if save_img or view_img:  # Add bbox to image
                        label = '%s%.2f' % (names[int(cls)],conf.item())
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=2)

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = 'mp4v'  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        print('Results saved to %s' % Path(out))
        if platform == 'darwin' and not opt.update:  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))
    
    import json 
    import numpy as np 

    input_json = opt.source
    with open(input_json) as file:
        parsed = json.load(file)
    image_list = parsed['images']
                            
    dict_7 = {'one_aspect': 0, 
            'two_aspects': 1,
            'three_aspects': 2,
            'front': 3,
            'back': 4,
            'right': 5,
            'left': 6}
    att = {v: k for k, v in dict_7.items()}                            
    # parsed_idx = 0
    # for images, outputs in zip(image_list, pred_list):  # batched inputs/results
    del_list = []
    for si, (image, output) in enumerate(zip(image_list, pred_list)): 
        if output is None or len(output) == 1:   # to distunguish images with no predictions                         
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
            del_list.append(si)
            # import pdb;pdb.set_trace()           
            continue
        # output[:, :4] = scale_coords(img.shape[2:], output[:, :4], im0.shape).round()
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
                # bbox = label_group[0][:4] # xyxy
                # cls = label_group[0][5]                        
                # if cls < 3:
                #     aspect = att[cls]
                #     direction = att[3]
                # else:
                #     aspect = att[2]
                #     direction = att[cls]
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
        
        if label_dict_list:
            image["labels"] = label_dict_list
        else:
            del_list.append(si)
            
    
    image_list = np.delete(image_list, del_list).tolist()
    parsed['images'] = image_list

    timestr = time.strftime("%Y_%m_%d_%H%M") + '_'
    input_name = input_json.split('/')[-1][:-5] + '_'
    path = os.path.join('csp', timestr + input_name + 'detection.json')
    with open(path, 'w+') as f:
        json.dump(parsed, f, indent=4) 
    print('Saved to', path)  

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov4-p5.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()

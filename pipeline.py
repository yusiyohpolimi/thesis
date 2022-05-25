import os
import argparse
import sys
sys.path.append("det2/")
import detect

def main(opt):
    
    label_file = detect.main(opt)
    os.system(
        "python3 -W ignore mapping.py " 
        + ' --calib-dir ' + opt.calib_dir
        + ' --database-dir ' + opt.database_dir
        + ' --gt-file ' + opt.gt_file
        + ' --label-file ' + label_file
    )
    print('Final map is saved to cwd.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='', help='initial weights path')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='test sizes')
    parser.add_argument('--iou', type=float, default=0.5, help='nms iou threshold')
    parser.add_argument('--conf', type=float, default=0.5, help='confidence score threshold')
    parser.add_argument('--anchors', action='store_true', help='use custom anchors')
    parser.add_argument('--model', type=str, default='R50', help='R50, X101')
    parser.add_argument('--json-path', type=str, default='default_inp.json', 
                        help='input json file with image paths and vehicle data')
    parser.add_argument('--cpu', action='store_true', help='use cpu for inference')
    parser.add_argument('--output', type=str, default='', help='folder to save detections')

    parser.add_argument('--calib-dir', type=str, default='/dtld_parsing/calibration/', 
                        help='calibration data directory')
    parser.add_argument('--database-dir', type=str, default='/multiverse/datasets/shared/DTLD/Berlin_disp', 
                        help='if all disparity images are extracted from the archive, no need to use this')                        
    parser.add_argument('--gt-file', type=str, default='/multiverse/datasets/shared/DTLD/v2.0/v2.0/DTLD_test.json', 
                        help='ground-truth json file')    

    opt = parser.parse_args()

    print('\nRunning commands: ')
    print(
        'python3 det2/detect.py'
        + ' --weights ' + opt.weights
        + ' --img ' + str(opt.img_size[0]) + ' ' + str(opt.img_size[1])
        + ' --iou ' + str(opt.iou)
        + ' --conf ' + str(opt.conf)
        + ' --model ' + opt.model
        + ' --json-path ' + opt.json_path
        + ' --output ' + opt.output
        + ' --anchors ' if opt.anchors else None
    )
    print(
        'python3 mapping.py'
        + ' --calib-dir ' + opt.calib_dir
        + ' --database-dir ' + opt.database_dir
        + ' --gt-file ' + opt.gt_file
    )
  
    main(opt)
      
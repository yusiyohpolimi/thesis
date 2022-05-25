#!/usr/bin/python3

import yaml
import os

def yolo_format(yml_file):
    
    obj_count = 0

    with open(yml_file, 'r') as stream:
        try:
            parsed = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)                

    # if yml_file == 'DTLD_train.yml':
    #     list_file = open('train.txt', 'w')
    
    # else:
    #     list_file = open('test.txt', 'w')

    for image in range(len(parsed)):
        img_path = parsed[image]['path']
        
        if os.path.exists(img_path[:-4] + 'txt'):
            os.remove(img_path[:-4] + 'txt')


        f = open(img_path[:-4] + 'txt', "w")
      
        # list_file.write(img_path + '\n')    
    
        for obj in range(len(parsed[image]['objects'])):
            dw = 1./2048
            dh = 1./1024
            x = parsed[image]['objects'][obj]['x']          
            y = parsed[image]['objects'][obj]['y']
            w = parsed[image]['objects'][obj]['width']
            h = parsed[image]['objects'][obj]['height']
            
            # if w < 10 or h < 10:
            #     continue

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

            xc = x + w/2
            yc = y + h/2
            xc = xc*dw
            yc = yc*dh
            w = w*dw
            h = h*dh
            obj_count += 1

            state = int(str(parsed[image]['objects'][obj]['class_id'])[-2])
            f.write(str(state) + " " + str(xc) + " " + str(yc) + " " + str(w) + " " + str(h) + '\n')
        f.close()
    stream.close()
    print(obj_count)

train_path = '/multiverse/datasets/shared/DTLD/DTLD_train.yml'
yolo_format(train_path)
print("Train txt files are created")

test_path = '/multiverse/datasets/shared/DTLD/DTLD_test.yml'
yolo_format(test_path)
print("Test txt files are created")
import json
import os

def yolo_format(json_file):
    obj_count = 0

    with open(json_file, 'r') as stream:        
        parsed = json.load(stream)

    for idx in range(len(parsed['images'])):
        images = parsed['images']
        img_path = images[idx]['image_path']  
        path_split = img_path.split('/')
        path_split[0] = '/multiverse/datasets/shared/DTLD'
        img_path = '/'.join(path_split)
        
        if os.path.exists(img_path[:-4] + 'txt'):
            os.remove(img_path[:-4] + 'txt')
        f = open(img_path[:-4] + 'txt', 'w')

        for obj in images[idx]['labels']:
            dw = 1./2048
            dh = 1./1024
            x = obj['x']
            y = obj['y']
            w = obj['w']
            h = obj['h']

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
            
            aspect = obj['attributes']['aspects']
            direction = obj['attributes']['direction']
            orientation = obj['attributes']['orientation']
            occ = obj['attributes']['occlusion']
            reflection = obj['attributes']['reflection']          
            
            if reflection == 'reflected' or occ == 'occluded' or aspect == 'four_aspects' \
                or orientation == 'horizontal' or aspect == 'unknown':
                continue            
            obj_count += 1         

            dict = {'one_aspect': '0', 
                    'two_aspects': '1',
                    'three_aspects': '2',
                    'front': '3',
                    'back': '4',
                    'right': '5',
                    'left': '6',}            

            state_asp = dict[aspect]
            state_dir = dict[direction]

            f.write(state_asp + " " + str(xc) + " " + str(yc)
                     + " " + str(w) + " " + str(h) + '\n' )
            f.write(state_dir + " " + str(xc) + " " + str(yc)
                     + " " + str(w) + " " + str(h) + '\n' )       
   
    print(obj_count)


train_path = '/multiverse/datasets/shared/DTLD/v2.0/v2.0/DTLD_train.json'
yolo_format(train_path)
print("Train txt files are created")

test_path = '/multiverse/datasets/shared/DTLD/v2.0/v2.0/DTLD_test.json'
yolo_format(test_path)
print("Test txt files are created")
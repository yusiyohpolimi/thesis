import os
import shutil

test_path = '/multiverse/datasets/shared/DTLD/test_folder'

if not os.path.exists(test_path):
    os.makedirs(test_path)

img_file = open('/multiverse/datasets/shared/DTLD/test.txt','r')
img_paths = img_file.read().split('\n')

for img in img_paths:
    if img:
        text = img[:-4] + 'txt'        
        new_path = test_path + '/' + img.split('/')[-1]
        text_path = test_path + '/' + img.split('/')[-1][:-4] + 'txt'
        
        if text_path == test_path + '/' + 'DE_BBBR667_2015-04-23_13-20-52-402607_k0.txt':
            continue
        # shutil.copyfile(img, new_path)
        shutil.copyfile(text, text_path)


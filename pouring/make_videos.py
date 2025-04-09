import os
import cv2
import numpy as np

subfix = ''

if not os.path.exists(f'stimuli/videos{subfix}'):
    os.makedirs(f'stimuli/videos{subfix}')

for folder in os.listdir(f'stimuli/sequences{subfix}'):
    if folder.startswith('.'):
        continue

    print(f'Making video for {folder}')
    img_array = []

    for file in sorted(os.listdir(f'stimuli/sequences{subfix}/' + folder), key=lambda x: float(x[:-4])):
        img = cv2.imread(f'stimuli/sequences{subfix}/{folder}/{file}')
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)
    print(len(img_array))

    out = cv2.VideoWriter(f'stimuli/videos{subfix}/{folder}.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 20, size)
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
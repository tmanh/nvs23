import cv2
import random
import numpy as np

from models.misc.degradation import *


def aug_moving(stacked_img,x,y,ph,pw):
    n,h,w = stacked_img.shape[:3]
    # print('n,h,w',n,h,w,x,y)
    step_x = random.randint(-1,1)
    step_y = random.randint(-1,1)
    # print('-------------')
    # print('vol',step_x,step_y)
    img_moving = []
    for idx in range(n):
        if idx!= n //2:
            dis = np.abs(n //2 -idx)
            if idx<n//2:
                dx =step_x* dis
                dy =step_y* dis
            else:
                dx = -step_x* dis
                dy = -step_y* dis

            # print(idx,dx,dy)
            bx = x + dx
            by = y + dy 
            bx = max(0,bx)
            bx = min(bx,w-1)

            by = max(0,by)
            by = min(by,h-1)

            # print('0',bx,by)
            pad_x = w - (bx + pw)
            if pad_x<0:
                bx = bx + pad_x 
            pad_y = h - (by + ph)
            if pad_y<0:
                by = by + pad_y

            # print(pad_x,pad_y,x,y,bx,by)
            img_crop = stacked_img[idx,by:by+ph,bx:bx+pw]
        else:
            img_crop = stacked_img[idx,y:y+ph,x:x+pw]
        # print('img_crop',img_crop.shape)
        img_moving.append(img_crop)
    return np.stack(img_moving,0) 


def overlapping(augmented):
    shift_x, shift_y = random.randint(-10, 10), random.randint(-10, 10)
    overlay = np.roll(augmented, shift=(shift_x, shift_y), axis=(0, 1))
    augmented = cv2.addWeighted(augmented, 0.7, overlay, 0.3, 0)
    return augmented


def test():

    path = '/scratch/antruong/workspace/render/RIEnhancer/submodules/DiffBIR/GT/_DSC8681.jpg'
    hr = cv2.imread(path)

    H, W = hr.shape[:2]
    coord_map = defineCoorMap(H, W)

    mask = defineHighlightArea(H, W, coord_map.copy())
    print(mask.shape)
        
    # print('mask',mask.max(),mask.min())
    lr_data = hr.copy()
    tmp = color_jet(lr_data)
    lr_data = lr_data * (1 - mask) + mask * tmp


    hr_data = hr.copy()
    lr_data, jpeg_quality,noise_level = process(lr_data, coord_map.copy())
    lr_data = reposition(lr_data, ratio=0.3)

    lr_data = overlapping(lr_data)

    print(lr_data.shape, lr_data.min(), lr_data.max())

    cv2.imwrite('degrad.png', lr_data.astype(np.uint8))



test()
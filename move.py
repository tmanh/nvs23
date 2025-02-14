import os
import cv2
import shutil
import numpy as np

root_dir = 'arkitscenes'
destination_folder = "p_arkitscenes"

os.makedirs(destination_folder, exist_ok=True)


all_objs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]


for index in range(len(all_objs)):
    scan_index = all_objs[index]
    inames = np.load(os.path.join(root_dir, scan_index, 'images.npy'))
    rgb_paths = [
        os.path.join(root_dir, scan_index, 'vga_wide', i.replace('png', 'jpg')) for i in inames
    ]

    depth_sens = [
        os.path.join(root_dir, scan_index, 'lowres_depth', i) for i in inames
    ]

    img = cv2.imread(rgb_paths[0])

    move = img.shape[0] == 640

    # bad = False
    # for d in depth_sens:
    #     raw_depth = cv2.imread(d, cv2.IMREAD_UNCHANGED) / 1000.0
    #     if raw_depth.max() <= 0:
    #         bad = True
    #         print('xxxx: scan_index')

    if move:
        src = os.path.join(root_dir, scan_index)
        dst = os.path.join(destination_folder, scan_index)
        shutil.move(os.path.join(root_dir, scan_index), os.path.join(destination_folder, scan_index))
        print(f"Folder moved from {src} to {dst}")

# print(all_objs)
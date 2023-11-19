import os
import torch
import torch.nn.functional as F

import cv2

import glob
import imageio
import numpy as np

from .util import get_image_to_tensor_balanced, load_pfm, ColorJitter
from .mask_generator import MaskGenerator


class DTU_Dataset(torch.utils.data.Dataset):
    def __init__(self, stage="train", opts=None):
        super().__init__()
        path = opts.path
        list_prefix = opts.list_prefix
        self.input_view_num = opts.input_view_num
        self.num_views = opts.num_views
        self.base_path = path
        self.stage = stage
        self.image_size = opts.image_size
        assert os.path.exists(self.base_path)
        self.train_path = os.path.join(path, f"{list_prefix}train.lst")
        self.val_path = os.path.join(path, f"{list_prefix}val.lst")
        self.test_path = os.path.join(path, f"{list_prefix}test.lst")
        if stage == "train":
            self.train = True
            file = os.path.join(path, f"{list_prefix}train.lst")
        elif stage in ["val", "test"]:
            self.train = False
            file = os.path.join(path, f"{list_prefix}val.lst")
        with open(file, "r") as f:
            objs = [x.strip() for x in f.readlines()]
        f.close()
        all_objs = list(objs)
        self.all_objs = all_objs
        self.image_to_tensor = get_image_to_tensor_balanced()
        print("Loading DVR dataset", self.base_path, "stage", stage, len(self.all_objs), "objs")

        self._coord_trans_world = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=torch.float32)

        self._coord_trans_cam = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=torch.float32)

        self.scale_factor = opts.scale_factor
        self.max_imgs = opts.max_imgs
        self.z_near = opts.z_near
        self.z_far = opts.z_far
        self.lindisp = False
        self.camera_path = opts.camera_path
        self.depth_path = opts.depth_path
        self.camera = np.load(self.camera_path, allow_pickle=True)
        self.colorjitter = ColorJitter()
        if self.stage == "test":
            self.test_view = list(map(int, opts.test_view.split()))

        self.prepare_for_training_depth = opts.train_depth_only

        self.mask_generator = MaskGenerator(300, 400, 1)

    def __len__(self):
        return len(self.all_objs) if self.stage != 'val' else len(self.all_objs) * 5

    def __getitem__(self, index):
        if self.stage == 'val':
            index = index % len(self.all_objs)

        scan_index = self.all_objs[index]

        ## read RGB image
        root_dir, rgb_paths, depth_paths = self.get_path_from(scan_index)

        if self.stage in ["test"]:
            # for test, we use the test_view as inputs and generate all the other views.
            all_indices = list(np.arange(len(rgb_paths)))
            sel_indices = list(self.test_view)
            all_indices = [x for x in all_indices if x not in sel_indices]
            sel_indices += all_indices
        else:
            # We randomly select num_views images, whhile the first input_view_num image for input.
            sample_num = min(self.num_views, self.max_imgs)
            sel_indices = torch.randperm(len(rgb_paths))[:sample_num]

        rgb_paths = [rgb_paths[i] for i in sel_indices]

        all_imgs = []
        all_poses = []
        all_poses_inverse = []
        all_raw_depths = []
        all_merge_depths = []
        focal = None
        raw_depth_max = 0
        fx, fy, cx, cy = 0.0, 0.0, 0.0, 0.0

        for idx, rgb_path in enumerate(rgb_paths):
            i = sel_indices[idx]
            img = imageio.imread(rgb_path)[..., :3]
            merge_depth, raw_depth = self.merge_depth_from_path(depth_paths[i], img)
            raw_depth_tensor = torch.tensor(raw_depth.copy())
            merge_depth_tensor = torch.tensor(merge_depth.copy())

            all_raw_depths.append(raw_depth_tensor)
            all_merge_depths.append(merge_depth_tensor)
            if raw_depth_tensor.max() > raw_depth_max:
                raw_depth_max = raw_depth_tensor.max()

            R = self.camera[i][0][:3, :3]
            T = self.camera[i][0][:3, 3]
            T = T / self.scale_factor ## we scale the word coordinate.
            K = self.camera[i][1]

            pose = np.eye(4, dtype=np.float32)
            pose[:3, :3] = R
            pose[:3, 3] = T
            pose =  torch.tensor(pose, dtype=torch.float32)

            fx += K[0, 0]
            fy += K[1, 1]
            cx += K[0, 2]
            cy += K[1, 2]
            img_tensor = self.image_to_tensor(img)
            pose_inverse = torch.inverse(pose)
            all_imgs += [img_tensor]
            all_poses += [pose]
            all_poses_inverse += [pose_inverse]

        all_raw_depths = [depth / self.scale_factor for depth in all_raw_depths]
        all_merge_depths = [depth / self.scale_factor for depth in all_merge_depths]

        fx /= len(rgb_paths)
        fy /= len(rgb_paths)
        cx /= len(rgb_paths)
        cy /= len(rgb_paths)

        focal = torch.tensor((fx, fy), dtype=torch.float32)
        c = torch.tensor((cx, cy), dtype=torch.float32)
        all_imgs = torch.stack(all_imgs)

        K = torch.zeros((4,4) , dtype=torch.float32)
        K[0, 0] = fx
        K[1, 1] = fy
        K[0, 2] = cx
        K[1, 2] = cy
        K[2, 2] = 1.0
        K[3, 3] = 1.0
        inverse_K = torch.inverse(K)

        if self.image_size is not None and all_imgs.shape[-2:] != self.image_size:
            scale = self.image_size[0] / all_imgs.shape[-2]
            K[0:2, 0:3] *= scale
            all_imgs = F.interpolate(all_imgs, size=self.image_size, mode="area")

        all_raw_depths = self.stack_depth_tensors(all_imgs, all_raw_depths)
        all_merge_depths = self.stack_depth_tensors(all_imgs, all_merge_depths)
        
        if self.train:
            all_imgs = self.colorjitter.apply_color_jitter(all_imgs)

        all_imgs = [all_imgs[i] for i in range(all_imgs.shape[0])]
        cameras = [{'focal': focal, 'P': all_poses[i], 'Pinv': all_poses_inverse[i], 'K': K, 'Kinv': inverse_K, 'c': c} for i in range(len(all_imgs))]

        augmented_masks = []
        height, width = all_raw_depths[0].shape[-2:]
        for _ in all_raw_depths:
            augmented_mask = self.mask_generator.sample()
            augmented_masks.append(torch.tensor(augmented_mask.reshape((1, height, width))))
        
        return {"idx":scan_index, "path": root_dir, "img_id": index, "images": all_imgs, "cameras": cameras, "depths": all_raw_depths, "mdepths": all_merge_depths, "augmented_mask": augmented_masks}

    def stack_depth_tensors(self, all_imgs, all_raw_depths):
        all_raw_depths = torch.stack(all_raw_depths, dim=0)
        all_raw_depths = F.interpolate(all_raw_depths.unsqueeze(0), size=all_imgs.shape[-2:], mode="nearest")
        all_raw_depths = all_raw_depths[0]
        return [all_raw_depths[i].unsqueeze(0) for i in range(all_raw_depths.shape[0])]

    def merge_depth_from_path(self, paths, image):
        depth_sen, _ = load_pfm(paths[0])
        depth_mvs, _ = load_pfm(paths[1])

        h2, w2 = depth_sen.shape
        depth_mvs = cv2.resize(depth_mvs, dsize=(w2, h2), interpolation=cv2.INTER_NEAREST)
        depth_mvs = cv2.medianBlur(depth_mvs, 3)

        mask_mvs = (depth_mvs > 0).astype(float)
        mask_sen = (depth_sen > 0).astype(float)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        object_mask = (gray > gray.min() + 10).astype(np.uint8)
        white_mask = (gray > gray.max() - 3).astype(np.uint8)
        white_mask = self.remove_small(white_mask)
        object_mask = cv2.medianBlur(object_mask, 3)

        diff = depth_mvs - depth_sen
        diff[depth_sen == 0] = -1000

        masked_diff = mask_mvs * diff

        merge_depth = np.zeros_like(depth_sen)
        merge_depth[mask_sen > 0] = depth_sen[mask_sen > 0]
        merge_depth[(mask_sen == 0) & (mask_mvs > 0) & (white_mask == 0)] = depth_mvs[(mask_sen == 0) & (mask_mvs > 0) & (white_mask == 0)]
        merge_depth[(masked_diff < -150) & (white_mask == 0) & (mask_mvs > 0) ] = depth_mvs[(masked_diff < -150) & (white_mask == 0) & (mask_mvs > 0) ]
        merge_depth[object_mask == 0] = 0

        return merge_depth, depth_sen
    
    def remove_small(self, binary_mask):
        # Set a minimum area threshold for small object removal
        min_area_threshold = 3000  # Adjust this value as needed

        # Find contours in the binary mask
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create an empty mask to draw the filtered contours
        filtered_mask = np.zeros_like(binary_mask)

        # Iterate through the contours and remove small objects
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= min_area_threshold:
                # Draw the contour on the filtered mask
                cv2.drawContours(filtered_mask, [contour], -1, 255, thickness=cv2.FILLED)

        return filtered_mask

    def get_path_from(self, scan_index):
        root_dir = os.path.join(self.base_path, "Rectified", scan_index)

        rgb_paths = os.path.join(root_dir, 'image', "*")
        rgb_paths = [
            x
            for x in glob.glob(rgb_paths)
            if x.endswith(".jpg") or x.endswith(".png")
        ]
        
        depth_sen_path = os.path.join(self.depth_path, scan_index, 'depth_map_*.pfm')
        depth_mvs_path = os.path.join(self.depth_path, scan_index, 'depth_map_*.pfm')

        depth_sens = glob.glob(depth_sen_path)
        depth_mvss = glob.glob(depth_mvs_path)

        depth_sens = sorted(depth_sens)
        depth_mvss = sorted(depth_mvss)

        depth_paths = []
        for x, y in zip(depth_sens, depth_mvss):
            depth_paths.append((x, y))

        rgb_paths = sorted(rgb_paths)
        return root_dir, rgb_paths, depth_paths
 
    def augment(self, inputs, flip_index, x, y, patch_size):
        inputs = inputs[:, y:y + patch_size, x:x + patch_size]

        if flip_index > 0:
            inputs = torch.flip(inputs, dims=[-2])
        return inputs

    def totrain(self, epoch=0):
        self.train = True
        with open(self.train_path, "r") as f:
            objs = [x.strip() for x in f.readlines()]
        f.close()
        all_objs = list(objs)
        self.all_objs = all_objs
        self.stage = "train"

    def toval(self, epoch=0):
        self.train = False
        with open(self.val_path, "r") as f:
            objs = [x.strip() for x in f.readlines()]
        f.close()
        all_objs = list(objs)
        self.all_objs = all_objs
        self.stage = "val"

    def totest(self, epoch=0):
        self.train = False
        with open(self.val_path, "r") as f:
            objs = [x.strip() for x in f.readlines()]
        f.close()
        all_objs = list(objs)
        self.all_objs = all_objs
        self.stage = "test"

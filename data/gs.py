import torch
import torch.nn.functional as F
import torchvision.transforms.functional as Ft

import os
import random
import cv2
import pickle
import os.path as osp
import numpy as np


def random_crop_arr(tgt, prd, image_size):
    crop_y = random.randrange(tgt.shape[0] - image_size + 1)
    crop_x = random.randrange(tgt.shape[1] - image_size + 1)
    return tgt[crop_y : crop_y + image_size, crop_x : crop_x + image_size], prd[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def random_crop(img, image_size):
    crop_y = random.randrange(img.shape[0] - image_size + 1)
    crop_x = random.randrange(img.shape[1] - image_size + 1)
    return img[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def random_hue_change(image1, image2, image3, image4, hue_range=(-20, 20)):
    """
    Randomly changes the hue of four images within the specified range.

    Args:
    - image1, image2, image3, image4 (numpy.ndarray): Input images in RGB format.
    - hue_range (tuple): Range of hue adjustment values (e.g., (-20, 20)).

    Returns:
    - tuple: Four images with the same adjusted hue.
    """
    # Generate a single random hue adjustment value
    hue_adjustment = np.random.randint(hue_range[0], hue_range[1])

    # Function to apply the hue change
    def apply_hue_change(image, hue_adjustment):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        hsv_image[:, :, 0] = (hsv_image[:, :, 0] + hue_adjustment) % 180  # Hue values range from 0 to 179 in OpenCV
        return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)

    # Apply the same hue adjustment to all four images
    adjusted_image1 = apply_hue_change(image1, hue_adjustment)
    adjusted_image2 = apply_hue_change(image2, hue_adjustment)
    adjusted_image3 = apply_hue_change(image3, hue_adjustment)
    adjusted_image4 = apply_hue_change(image4, hue_adjustment)

    return adjusted_image1, adjusted_image2, adjusted_image3, adjusted_image4


class DiffData:
    def __init__(
            self,
            file_list: str,
            out_size: int,
            val: bool
        ):
        """
        Args:
            image_dir (string): Path to the directory with images.
            mask_dir (string, optional): Path to the directory with masks (for segmentation tasks).
            transform (callable, optional): Optional transform to be applied on a sample (image).
            target_transform (callable, optional): Optional transform to be applied on the target (mask or label).
        """

        self.path = file_list
        self.file_list = file_list
        self.metadata = osp.join(file_list, 'metadata.pkl')
        self.val = val
        self.image_size = out_size
        self.scan()

    def scan(self):
        with open(self.metadata, 'rb') as f:
            metadata = pickle.load(f)

        self.e_gts = []
        self.gts = []
        self.renders = []
        self.masks = []
        self.depths = []
        kd = 'all_test_view_dict' if self.val else 'all_train_view_dict'
        for k in metadata[kd].keys():
            view = metadata[kd][k]
            for v in view:
                if not osp.exists(v):
                    v = v.replace('/scratch/guachen/datasets/wildrgbd/sparse/diff_dataset_test', '/scratch/antruong/workspace/myspace/datasets/diff_dataset')

                render = osp.join(v, 'render.png')
                mask = osp.join(v, 'mask.png')
                gt_enhanced = osp.join(v, 'gt_enhanced.png')
                gt = osp.join(v, 'gt.png')
                depth = osp.join(v, 'depth.npy')
                    
                self.renders.append(render)
                self.masks.append(mask)
                self.e_gts.append(gt_enhanced)
                self.gts.append(gt)
                self.depths.append(depth)
    
    def with_transform(self, preprocess_train):
        self.transform = preprocess_train
        return self

    def __len__(self):
        return len(self.e_gts)

    def __getitem__(self, idx):
        while not os.path.exists(self.renders[idx]) or not os.path.exists(self.e_gts[idx]):
            idx = np.random.randint(0, len(self))
        
        prd = cv2.imread(self.renders[idx]) / 255.0 * 2.0 - 1.0
        tgt = cv2.imread(self.e_gts[idx]) / 255.0 * 2.0 - 1.0
        prompt = ""

        tgt, prd = random_crop_arr(tgt, prd, self.image_size)

        return tgt, prd, prompt
    

class DiffData2:
    def __init__(
            self,
            file_list: str,
            out_size: int,
            val: bool
        ):
        """
        Args:
            image_dir (string): Path to the directory with images.
            mask_dir (string, optional): Path to the directory with masks (for segmentation tasks).
            transform (callable, optional): Optional transform to be applied on a sample (image).
            target_transform (callable, optional): Optional transform to be applied on the target (mask or label).
        """

        self.path = file_list
        self.file_list = file_list
        self.metadata = osp.join(file_list, 'metadata.pkl')
        self.val = val
        self.image_size = out_size
        self.scan()

    def scan(self):
        with open(self.metadata, 'rb') as f:
            metadata = pickle.load(f)

        kd = 'train' if self.val else 'eval'
        self.scenes = metadata[kd]
    
    def with_transform(self, preprocess_train):
        self.transform = preprocess_train
        return self

    def __len__(self):
        return len(self.scenes) * 50 if not self.val else len(self.scenes)

    def read_data_train(self, idx):
        scene = self.scenes[idx]
        views = [osp.join(scene, d) for d in os.listdir(scene)]

        subviews = random.sample(views, k=3)
        main = subviews[0]
        ref1 = subviews[1]
        ref2 = subviews[2]

        render = osp.join(main, 'render.png')
        gt = osp.join(main, 'gt_enhanced.png')
        ref1 = osp.join(ref1, 'gt_enhanced.png')
        ref2 = osp.join(ref2, 'gt_enhanced.png')

        render = cv2.imread(render)
        ref1 = cv2.imread(ref1)
        ref2 = cv2.imread(ref2)
        gt = cv2.imread(gt)

        gt = cv2.resize(gt, (512, 512), interpolation=cv2.INTER_LANCZOS4)
        render = cv2.resize(render, (512, 512), interpolation=cv2.INTER_LANCZOS4)
        ref1 = cv2.resize(ref1, (512, 512), interpolation=cv2.INTER_LANCZOS4)
        ref2 = cv2.resize(ref2, (512, 512), interpolation=cv2.INTER_LANCZOS4)

        return random_hue_change(render, gt, ref1, ref2)

    def __getitem__(self, idx):
        prd, tgt, ref1, ref2 = self.read_data_train(idx % len(self.scenes))
        
        prd = prd / 255.0 * 2.0 - 1.0
        tgt = tgt / 255.0 * 2.0 - 1.0
        ref1 = ref1 / 255.0 * 2.0 - 1.0
        ref2 = ref2 / 255.0 * 2.0 - 1.0
        
        prompt = ""

        tgt, prd = random_crop_arr(tgt, prd, self.image_size)
        ref1 = random_crop(ref1, self.image_size)
        ref2 = random_crop(ref2, self.image_size)

        return tgt, prd, ref1, ref2, prompt
    

class DiffData3:
    def __init__(
            self,
            file_list: str,
            val: bool
        ):
        """
        Args:
            image_dir (string): Path to the directory with images.
            mask_dir (string, optional): Path to the directory with masks (for segmentation tasks).
            transform (callable, optional): Optional transform to be applied on a sample (image).
            target_transform (callable, optional): Optional transform to be applied on the target (mask or label).
        """

        self.path = file_list
        self.file_list = file_list
        self.metadata = osp.join(file_list, 'metadata.pkl')
        self.val = val
        self.scan()

        self.H = 512
        self.W = 384

        self.n_samples = 3

    def scan(self):
        self.scenes = []
        
        scenes = sorted([osp.join(self.path, d) for d in os.listdir(self.path)])
        for scene in scenes:
            # if 'apple_002' not in scene:
            #     continue
            sd = {
                'depth': osp.join(scene, 'depths.npy'),
                'intrinsic': osp.join(scene, 'intrinsic.npy'),
                'pose': osp.join(scene, 'pose.npy'),
                'color': osp.join(scene, 'colors.npy')
            }
            self.scenes.append(sd)
    
    def with_transform(self, preprocess_train):
        self.transform = preprocess_train
        return self

    def __len__(self):
        return len(self.scenes) if not self.val else int(len(self.scenes) * 0.05)

    def to_wild_pose_path(self, path):
        directory, name = os.path.split(path)
        directory = directory.replace(
            'dust3r_pcd_rendered_images/wildrgbd/diff_dataset_train',
            'wildrgbd/sparse/wildrgbd_sparse_8'
        )
        return osp.join(directory, f'train/poses/{name}.npz')
    
    def read_wild_pose(self, path):
        tmp = np.load(path)

        K = np.eye(4, 4)
        K[:3, :3] = tmp['camera_intrinsics']
        Rt = tmp['camera_pose']

        return K, Rt
    
    def read_data_with_idx(self, scene, idxs):
        colors = np.load(scene['color'])[idxs]
        depths = np.load(scene['depth'])[idxs]
        Ks = np.load(scene['intrinsic'])[idxs]
        Rts = np.load(scene['pose'])[idxs]

        return colors, depths, Ks, Rts

    def read_data_train(self, idx):
        scene = self.scenes[idx]
        
        idxs = list(np.arange(8))
        idxs = random.sample(idxs, k=self.n_samples)
        colors, depths, Ks, Rts = self.read_data_with_idx(scene, idxs)

        colors = torch.tensor(colors).permute(0, 3, 1, 2)
        depths = torch.tensor(depths).unsqueeze(1)

        H, W = colors.shape[-2:]
        Ks = torch.tensor(Ks)
        Ks[0, 0, :] = self.W / W * Ks[0, 0, :]
        Ks[0, 1, :] = self.H / H * Ks[0, 1, :]
        Rts = torch.tensor(Rts)

        return colors[:1], colors[1:], depths[:1], depths[1:], Ks[0], Rts[:1], Rts[1:]

    def __getitem__(self, idx):
        dst_cs, src_cs, dst_ds, src_ds, K, dst_Rts, src_Rts = self.read_data_train(idx % len(self.scenes))

        src_ds = F.interpolate(src_ds, size=(self.H, self.W), mode='nearest')
        dst_ds = F.interpolate(dst_ds, size=(self.H, self.W), mode='nearest')
        src_cs = F.interpolate(src_cs, size=(self.H, self.W), mode='bilinear', align_corners=True, antialias=True)
        dst_cs = F.interpolate(dst_cs, size=(self.H, self.W), mode='bilinear', align_corners=True, antialias=True)

        hue_adjustment = torch.FloatTensor(1).uniform_(-0.1, 0.1).item()
        src_cs = Ft.adjust_hue(src_cs, hue_adjustment)
        dst_cs = Ft.adjust_hue(dst_cs, hue_adjustment)

        return dst_cs, src_cs, dst_ds, src_ds, K, dst_Rts, src_Rts
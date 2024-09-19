import os
import cv2
import torch
import argparse
import time
import glob

import numpy as np
import os.path as osp
import torch.nn.functional as F


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.sys.path.append(os.path.abspath(os.path.join(BASE_DIR, "dust3r")))
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.device import to_numpy
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from utils.dust3r_utils import  compute_global_alignment, storePly, save_colmap_cameras, save_colmap_images
from dust3r.utils.image import _resize_pil_image

import PIL
import torchvision.transforms as tvf
from PIL.ImageOps import exif_transpose


def load_images(images, shape=(384, 512)):
    imgs = []
    for path in images:
        if not path.endswith(('.jpg', '.jpeg', '.png', '.JPG')):
            continue
        img = PIL.Image.open(path).convert('RGB')
        W1, H1 = img.size
        img = img.resize(shape, PIL.Image.Resampling.LANCZOS)

        ImgNorm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        imgs.append(dict(img=ImgNorm(img)[None], true_shape=np.int32(
            [img.size[::-1]]), idx=len(imgs), instance=str(len(imgs))))

    print(f' (Found {len(imgs)} images)')
    return imgs, (W1, H1)


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_size", type=int, default=512, choices=[512, 224], help="image size")
    parser.add_argument("--model_path", type=str, default="dust3r/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth", help="path to the model weights")
    parser.add_argument("--device", type=str, default='cuda', help="pytorch device")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--schedule", type=str, default='linear')
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--niter", type=int, default=300)
    parser.add_argument("--focal_avg", action="store_true")
    parser.add_argument("--llffhold", type=int, default=2)
    parser.add_argument("--n_views", type=int, default=8)
    parser.add_argument("--img_base_path", type=str, default="wildrgb")

    return parser


def get_image_list(path):
    # List of image file extensions to look for
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.gif', '*.tiff']

    # Get a list of all images in the folder
    images = []
    for extension in image_extensions:
        images.extend(glob.glob(os.path.join(path, extension)))
    return images


#--------------------------------------------------
if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    model_path = args.model_path
    device = args.device
    batch_size = args.batch_size
    schedule = args.schedule
    lr = args.lr
    niter = args.niter
    n_views = args.n_views
    img_base_path = args.img_base_path

    model = AsymmetricCroCo3DStereo.from_pretrained(model_path).to(device)

    scenes = [osp.join(img_base_path, d) for d in os.listdir(img_base_path)]
    for scene_path in scenes:
        print(scene_path)
        if 'apple_002' not in scene_path:
            continue
        if not osp.isdir(scene_path):
            continue
        # if osp.exists(os.path.join(scene_path, 'depths.npy')) or not osp.isdir(scene_path):
        #     continue
            
        view_folders = sorted([osp.join(scene_path, d) for d in os.listdir(scene_path) if osp.isdir(osp.join(scene_path, d))])
        views = sorted([osp.join(d, 'gt_enhanced.png') for d in view_folders])
        images, (H, W) = load_images(views)
        print(f"{scene_path}: ori_size", (H, W))

        start_time = time.time()
        pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
        output = inference(pairs, model, args.device, batch_size=batch_size)
        scene = global_aligner(output, device=args.device, mode=GlobalAlignerMode.PointCloudOptimizer)
        loss = compute_global_alignment(scene=scene, init="mst", niter=niter, schedule=schedule, lr=lr, focal_avg=args.focal_avg)
        scene = scene.clean_pointcloud()
        imgs = [cv2.resize(img, dsize=(W, H), interpolation=cv2.INTER_LANCZOS4) for img in scene.imgs]
        imgs = np.array(imgs)
        focals = scene.get_focals()
        poses = to_numpy(scene.get_im_poses())
        pts3d = to_numpy(scene.get_pts3d())
        scene.min_conf_thr = float(scene.conf_trf(torch.tensor(1.0)))
        confidence_masks = to_numpy(scene.get_masks())
        intrinsics = to_numpy(scene.get_intrinsics())
        depths = [
            F.interpolate(
                d.unsqueeze(0).unsqueeze(0), size=(H, W), mode='nearest'
            ).squeeze(0).squeeze(0).detach().cpu().numpy() for d in scene.get_depthmaps()]
        depths = np.array(depths)
        ##########################################################################################################################################################################################
        end_time = time.time()
        print(f"Time taken for {n_views} views: {end_time-start_time} seconds")

        new_intrinsics = np.eye(4, 4).reshape((1, 4, 4)).repeat(intrinsics.shape[0], 0)
        new_intrinsics[:, :3, :3] = intrinsics
        new_intrinsics[:, 0, :] = (W / 384) * new_intrinsics[:, 0, :]
        new_intrinsics[:, 1, :] = (H / 512) * new_intrinsics[:, 1, :]

        # save
        np.save(os.path.join(scene_path, 'pose.npy'), poses)
        np.save(os.path.join(scene_path, 'intrinsic.npy'), new_intrinsics)
        np.save(os.path.join(scene_path, 'depths.npy'), depths)
        np.save(os.path.join(scene_path, 'colors.npy'), imgs)

import os
import cv2
import torch
import argparse
import time
import glob

import pickle

import numpy as np
import os.path as osp
import torch.nn.functional as F


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.sys.path.append(os.path.abspath(os.path.join(BASE_DIR, "dust3r")))
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


from demo import get_3D_model_from_scene
from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.device import to_numpy
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from utils.dust3r_utils import  compute_global_alignment, lora3d_compute_global_alignment, storePly, save_colmap_cameras, save_colmap_images
from dust3r.utils.image import _resize_pil_image

import PIL
import torchvision.transforms as tvf
from PIL.ImageOps import exif_transpose


def load_images(images, size=512):
    imgs = []
    for path in images:
        if not path.endswith(('.jpg', '.jpeg', '.png', '.JPG')):
            continue

        img = PIL.Image.open(path).convert('RGB')
        width, height = img.size
        if width > height:
            new_width = size
            new_height = int((height / width) * size)
        else:
            new_height = size
            new_width = int((width / height) * size)
        new_width = new_width // 16 * 16
        new_height = new_height // 16 * 16
        shape = (new_width, new_height)

        img = img.resize(shape, PIL.Image.Resampling.LANCZOS)

        ImgNorm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        imgs.append(dict(img=ImgNorm(img)[None], true_shape=np.int32(
            [img.size[::-1]]), idx=len(imgs), instance=str(len(imgs))))

    print(f' (Found {len(imgs)} images)')
    return imgs


def filter_images(path, images):
    if 'arkitscenes_processed' not in path:
        return [images]
    
    first_time = -1
    list_chunks = []
    chunk = []
    for i, image in enumerate(images):
        curr_time = float(os.path.splitext(os.path.basename(image))[0].split('_')[1])
        
        if first_time == -1:
            first_time = curr_time
            chunk.append(image)
        elif curr_time - first_time < 1.75:
            chunk.append(image)
        elif i == len(images) - 1:
            list_chunks.append(chunk.copy())
            first_time = curr_time
            chunk = []
        else:
            list_chunks.append(chunk.copy())
            first_time = curr_time
            chunk = []

    list_chunks = [c for c in list_chunks if len(c) > 5]

    return list_chunks


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


def scan(path):
    if 'arkitscenes_processed' in path:
        return [osp.join(path, d, 'vga_wide') for d in os.listdir(path)]
    elif 'DTU' in path:
        return [osp.join(path, d, 'image') for d in os.listdir(path)]
    return [path]


def get_image_files(folder_path):
    # Define supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
    
    # List all files in the folder
    all_files = os.listdir(folder_path)
    
    # Filter for image files
    image_files = [os.path.join(folder_path, f) for f in all_files if os.path.splitext(f)[1].lower() in image_extensions]
    
    return image_files


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
    # img_base_path = 'datasets/arkitscenes_processed'
    img_base_path = 'datasets/example3'

    model = AsymmetricCroCo3DStereo.from_pretrained(model_path).to(device)

    scenes = scan(img_base_path)
    scenes = sorted(scenes)
    for scene_path in scenes:
        if not osp.isdir(scene_path):
            continue

        views = sorted(get_image_files(scene_path))
        views = filter_images(img_base_path, views)
        
        # with open(os.path.join(scene_path, 'meta.pkl'), 'wb') as f:
        #     pickle.dumps(views)

        for idx, vs in enumerate(views):
            if os.path.exists(
                os.path.join(scene_path, f'{idx}', 'pose.npy')
            ):
                continue

            images = load_images(vs)
            start_time = time.time()
            pairs = make_pairs(
                images, scene_graph='swin', prefilter=None, symmetrize=True
            )
            output = inference(pairs, model, args.device, batch_size=batch_size)
            scene = global_aligner(output, device=args.device, mode=GlobalAlignerMode.PointCloudOptimizer)
            
            # compute_global_alignment(scene=scene, init="mst", niter=niter, schedule=schedule, lr=lr, focal_avg=args.focal_avg)
            loss = lora3d_compute_global_alignment(scene=scene, init="mst", niter=niter, schedule=schedule, lr=lr, focal_avg=args.focal_avg)
                
            outfile = get_3D_model_from_scene(
                outdir='output', silent=False, as_pointcloud=True, scene=scene, clean_depth=True)
            print('here')
            exit()
            scene = scene.clean_pointcloud()
            imgs = np.array(scene.imgs)
            focals = scene.get_focals()
            poses = to_numpy(scene.get_im_poses())
            pts3d = to_numpy(scene.get_pts3d())
            scene.min_conf_thr = float(scene.conf_trf(torch.tensor(1.0)))
            confidence_masks = to_numpy(scene.get_masks())
            intrinsics = to_numpy(scene.get_intrinsics())
            depths = [
                d.squeeze(0).squeeze(0).detach().cpu().numpy() for d in scene.get_depthmaps()]
            depths = np.array(depths)
                
            ##########################################################################################################################################################################################
            end_time = time.time()
            print(f"Time taken for {n_views} views: {end_time-start_time} seconds")

            new_intrinsics = np.eye(4, 4).reshape((1, 4, 4)).repeat(intrinsics.shape[0], 0)
            new_intrinsics[:, :3, :3] = intrinsics

            # save
            os.makedirs(os.path.join(scene_path, f'{idx}'), exist_ok=True)
            np.save(os.path.join(scene_path, f'{idx}', 'pose.npy'), poses)
            np.save(os.path.join(scene_path, f'{idx}', 'intrinsic.npy'), new_intrinsics)
            np.save(os.path.join(scene_path, f'{idx}', 'depths.npy'), depths)

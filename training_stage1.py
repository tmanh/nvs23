import os
from argparse import ArgumentParser
import warnings

import cv2
import numpy as np

from omegaconf import OmegaConf
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from accelerate import Accelerator
from accelerate.utils import set_seed
from tqdm import tqdm
import kornia

from data.multi import MultiDataset, MultiDataLoader
from models.losses.GlobalPercLoss import radiov2_5_loss
from models.losses.synthesis import *
from models.losses.cobi import ContextualBilateralLoss, ContextualLoss
from models.synthesis.fwd import FWD
from models.synthesis.lightformer import LightFormer
from models.synthesis.deepblendplus import DeepBlendingPlus

from utils.common import instantiate_from_config


def main(args) -> None:
    # Setup accelerator:
    accelerator = Accelerator(
        split_batches=True,
    )
    set_seed(231)
    device = accelerator.device
    cfg = OmegaConf.load(args.config)

    # Setup an experiment folder:
    if accelerator.is_local_main_process:
        exp_dir = cfg.train.exp_dir
        os.makedirs(exp_dir, exist_ok=True)
        ckpt_dir = os.path.join(exp_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        print(f"Experiment directory created at {exp_dir}")

    # Create model:
    renderer = FWD(cfg)
    # renderer = DeepBlendingPlus(cfg)
    if cfg.train.resume and os.path.exists(cfg.train.resume):
        renderer.load_state_dict(torch.load(cfg.train.resume, map_location="cpu"), strict=True)
        if accelerator.is_local_main_process:
            print(f"strictly load weight from checkpoint: {cfg.train.resume}")
    else:
        if accelerator.is_local_main_process:
            print("initialize from scratch")
    
    # Setup optimizer:
    opt = torch.optim.AdamW(
        [p for p in renderer.parameters() if p.requires_grad],  # Only trainable params
        lr=cfg.train.learning_rate,
        weight_decay=0
    )
    
    # Setup data:
    if 'MultiDataLoader' in cfg.dataset.train.target:
        loader = instantiate_from_config(cfg.dataset.train)
    else:
        dataset = instantiate_from_config(cfg.dataset.train)
        loader = DataLoader(
            dataset=dataset, batch_size=cfg.train.batch_size,
            num_workers=cfg.train.num_workers,
            shuffle=True, drop_last=True
        )

    if 'MultiDataLoader' in cfg.dataset.val.target:
        val_loader = instantiate_from_config(cfg.dataset.val)
    else:
        val_dataset = instantiate_from_config(cfg.dataset.val)
        val_loader = DataLoader(
            dataset=val_dataset, batch_size=cfg.train.batch_size,
            num_workers=cfg.train.num_workers,
            shuffle=False, drop_last=False
        )

    ploss = radiov2_5_loss()

    # Prepare models for training:
    renderer.to_train().to(device)
    renderer, opt, loader, val_loader = accelerator.prepare(renderer, opt, loader, val_loader)
    pure_renderer = accelerator.unwrap_model(renderer)

    # Variables for monitoring/logging purposes:
    global_step = 0
    max_steps = cfg.train.train_steps
    step_loss = []
    epoch = 0
    epoch_loss = []

    if accelerator.is_local_main_process:
        writer = SummaryWriter(exp_dir)
        print(f"Training for {max_steps} steps...")

    while global_step < max_steps:
        total_l1 = 0
        total_lp = 0
        pbar = tqdm(iterable=None, disable=not accelerator.is_local_main_process, unit="batch", total=len(loader))
        for local_step, (dst_cs, src_cs, dst_ds, src_ds, K, dst_Rts, src_Rts) in enumerate(loader):
            dst_ds = dst_ds.float().to(device)
            dst_cs = dst_cs.float().to(device)
            src_cs = src_cs.float().to(device)
            src_ds = src_ds.float().to(device)
            K = K.float().to(device)
            dst_Rts = dst_Rts.float().to(device)
            src_Rts = src_Rts.float().to(device)

            pred, mask, warp = renderer(
                src_ds, src_cs,
                K,
                src_Rts, torch.inverse(src_Rts), 
                dst_Rts, torch.inverse(dst_Rts),
                visualize=True
            )
            dst_cs = dst_cs.squeeze(1)

            if global_step % 250 == 0:
                x = dst_cs[0].permute(1, 2, 0) * 255
                x = x.clamp(0, 255)
                x = x.detach().cpu().numpy().astype(np.uint8)
                cv2.imwrite('output/c_tgt.png', cv2.cvtColor(x, cv2.COLOR_RGB2BGR))
                
                x = pred[0].permute(1, 2, 0) * 255
                x = x.clamp(0, 255)
                x = x.detach().cpu().numpy().astype(np.uint8)
                cv2.imwrite('output/c_prd.png', cv2.cvtColor(x, cv2.COLOR_RGB2BGR))

                lw = warp * 255
                lw = lw.clamp(0, 255.0)
                src_cs = src_cs * 255
                src_cs = src_cs.clamp(0, 255.0)
                for k in range(lw.shape[1]):
                    out = lw[0, k].permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)
                    cv2.imwrite(f'output/c_prj{k}.png', cv2.cvtColor(out, cv2.COLOR_RGB2BGR))

                    out = src_cs[0, k].permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)
                    cv2.imwrite(f'output/c_src{k}.png', cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
                # input()
                
            kernel = torch.ones((5, 5), device=mask.device)
            mask = kornia.morphology.closing(mask, kernel).detach()

            loss_l1 = F.l1_loss(pred * mask, dst_cs * mask)
            loss_p = 0.05 * ploss(pred * mask, dst_cs * mask)
            loss = loss_l1 + loss_p

            opt.zero_grad()
            accelerator.backward(loss)

            # for name, param in renderer.named_parameters():
            #     if param.grad is None:
            #         print(name)
            # exit()

            opt.step()
            accelerator.wait_for_everyone()

            global_step += 1
            step_loss.append(loss.item())
            epoch_loss.append(loss.item())
            pbar.update(1)

            total_l1 += loss_l1.item()
            total_lp += loss_p.item()
            pbar.set_description(f"Global Step: {global_step:07d}, L1: {(total_l1 / (local_step + 1)):.6f}, P: {(total_lp / (local_step + 1)):.6f}")

            # Log loss values:
            if global_step % cfg.train.log_every == 0:
                # Gather values from all processes
                avg_loss = accelerator.gather(torch.tensor(step_loss, device=device).unsqueeze(0)).mean().item()
                step_loss.clear()
                if accelerator.is_local_main_process:
                    writer.add_scalar("train/loss_step", avg_loss, global_step)

            # Save checkpoint:
            if global_step % cfg.train.ckpt_every == 0:
                if accelerator.is_local_main_process:
                    checkpoint = pure_renderer.state_dict()
                    ckpt_path = f"{ckpt_dir}/{global_step:07d}.pt"
                    torch.save(checkpoint, ckpt_path)

            accelerator.wait_for_everyone()

            if global_step == max_steps:
                break
        
        pbar.close()
        epoch += 1
        avg_epoch_loss = accelerator.gather(torch.tensor(epoch_loss, device=device).unsqueeze(0)).mean().item()
        epoch_loss.clear()
        if accelerator.is_local_main_process:
            writer.add_scalar("train/loss_epoch", avg_epoch_loss, global_step)

    if accelerator.is_local_main_process:
        print("done!")
        writer.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args)

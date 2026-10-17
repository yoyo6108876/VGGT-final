"""
distill2/data.py

直接讀 teacher_dataset_diverse 的 Dataset。
每個 scene 目錄含：meta.json, images/, depth.npy, depth_conf.npy,
extrinsics.npy, intrinsics.npy, pose_enc.npy, point_mask.npy。

回傳的 batch（經 DataLoader 後 stack 成 [B,S,...]）：
  images      [S,3,H,W] float [0,1]
  depth       [S,H,W]   teacher 深度（已 scale 正規化）
  depth_conf  [S,H,W]
  pose_enc    [S,9]     正規化後（first sampled frame = identity）重算的目標
  point_mask  [S,H,W]   bool
"""

import json
import random
from pathlib import Path

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

from vggt.utils.geometry import closed_form_inverse_se3
from vggt.utils.pose_enc import extri_intri_to_pose_encoding

_REQUIRED = ["meta.json", "depth.npy", "depth_conf.npy",
             "extrinsics.npy", "intrinsics.npy", "pose_enc.npy", "point_mask.npy"]


def _list_scenes(split_dir: Path):
    scenes = []
    if not split_dir.exists():
        return scenes
    for d in sorted(split_dir.iterdir()):
        if d.is_dir() and all((d / f).exists() for f in _REQUIRED):
            scenes.append(d)
    return scenes


def _normalize(extr, intr, depth, world_points, point_mask, image_hw):
    """
    first-frame canonical + scale 正規化（與舊 pipeline 完全等價），回傳重算的 pose_enc。
    extr:[S,3,4] intr:[S,3,3] depth:[S,H,W] world_points:[S,H,W,3]|None mask:[S,H,W] (torch)
    注意：scale 只影響 translation/depth；rotation 目標與 scale 無關。
    """
    S = extr.shape[0]
    homog = torch.cat([extr, torch.zeros(S, 1, 4)], dim=1)  # [S,4,4]
    homog[:, 3, 3] = 1.0
    inv0 = closed_form_inverse_se3(homog[:1])  # [1,4,4] cam0->world
    new_extr = torch.matmul(homog, inv0)[:, :3, :].clone()  # [S,3,4] frame0->identity

    # scale：把 teacher world_points 轉到 frame0 相機座標後取平均範數（同舊 pipeline）
    scale = torch.tensor(1.0)
    if world_points is not None:
        R0 = extr[0, :3, :3]; t0 = extr[0, :3, 3]
        wnew = world_points @ R0.T + t0          # [S,H,W,3]
        m = point_mask.bool()
        if m.any():
            scale = wnew[m].norm(dim=-1).mean().clamp(min=1e-6, max=1e6)

    new_extr[:, :3, 3] = new_extr[:, :3, 3] / scale
    depth = depth / scale

    pose_enc = extri_intri_to_pose_encoding(
        new_extr.unsqueeze(0), intr.unsqueeze(0),
        image_size_hw=image_hw, pose_encoding_type="absT_quaR_FoV",
    )[0]  # [S,9]
    return pose_enc, depth


class TeacherDataset(Dataset):
    def __init__(self, data_root, split="train", img_per_seq=4, length=2000, img_size=518,
                 gt_pose=False):
        self.split = split
        self.img_per_seq = img_per_seq
        self.length = length if split == "train" else max(1, length // 5)
        self.img_size = img_size
        self.gt_pose = gt_pose   # True：pose 目標改用 CO3D GT 外參（make_gt_labels.py 產生）
        self.scenes = _list_scenes(Path(data_root) / split)
        if not self.scenes:
            raise RuntimeError(f"找不到任何 scene：{Path(data_root)/split}")
        print(f"[TeacherDataset:{split}] {len(self.scenes)} scenes")

    def __len__(self):
        return self.length

    def _load_img(self, path):
        im = cv2.imread(str(path))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return torch.from_numpy(im).permute(2, 0, 1)  # [3,H,W]

    def __getitem__(self, idx):
        scene = random.choice(self.scenes) if self.split == "train" else self.scenes[idx % len(self.scenes)]
        meta = json.load(open(scene / "meta.json", encoding="utf-8"))
        n = meta["num_images"]
        k = min(self.img_per_seq, n)
        if self.split == "train":
            ids = sorted(random.sample(range(n), k))
        else:
            ids = list(np.linspace(0, n - 1, k).astype(int))

        depth_all = np.load(scene / "depth.npy", mmap_mode="r")
        conf_all = np.load(scene / "depth_conf.npy", mmap_mode="r")
        gt_path = scene / "gt_extrinsics.npy"
        if self.gt_pose and gt_path.exists():
            extr_all = np.load(gt_path, mmap_mode="r")   # 旋轉/平移目標 = CO3D GT
        else:
            extr_all = np.load(scene / "extrinsics.npy", mmap_mode="r")
        intr_all = np.load(scene / "intrinsics.npy", mmap_mode="r")
        mask_all = np.load(scene / "point_mask.npy", mmap_mode="r")
        wp_path = scene / "world_points.npy"
        wp_all = np.load(wp_path, mmap_mode="r") if wp_path.exists() else None

        imgs, depths, confs, extrs, intrs, masks, wps = [], [], [], [], [], [], []
        for i in ids:
            imgs.append(self._load_img(scene / "images" / meta["image_paths"][i]))
            depths.append(torch.from_numpy(np.asarray(depth_all[i], dtype=np.float32)))
            confs.append(torch.from_numpy(np.asarray(conf_all[i], dtype=np.float32)))
            extrs.append(torch.from_numpy(np.asarray(extr_all[i], dtype=np.float32)))
            intrs.append(torch.from_numpy(np.asarray(intr_all[i], dtype=np.float32)))
            masks.append(torch.from_numpy(np.asarray(mask_all[i])).bool())
            if wp_all is not None:
                wps.append(torch.from_numpy(np.asarray(wp_all[i], dtype=np.float32)))

        images = torch.stack(imgs)      # [S,3,H,W]
        depth = torch.stack(depths)     # [S,H,W]
        conf = torch.stack(confs)
        extr = torch.stack(extrs)       # [S,3,4]
        intr = torch.stack(intrs)
        mask = torch.stack(masks)
        world_points = torch.stack(wps) if wps else None
        H, W = images.shape[-2:]

        pose_enc, depth = _normalize(extr, intr, depth, world_points, mask, (H, W))

        return {
            "images": images, "depth": depth, "depth_conf": conf,
            "pose_enc": pose_enc, "point_mask": mask,
        }

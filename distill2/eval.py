"""
distill2/eval.py

評估 student vs teacher soft labels：rotation error / RRA@15 / depth abs_rel 等。
也提供 build_model / load_student 供 train.py 重用。
"""

import argparse
import json
from pathlib import Path

import numpy as np
import cv2
import torch

from vggt.models.vggt_student import VGGTStudent
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


def build_model(meta, device):
    """依 meta 建立 student（與 build_student.py 存的架構一致）。"""
    embed_dim = meta.get("embed_dim", 1024)
    return VGGTStudent(
        img_size=518, patch_size=14,
        embed_dim=embed_dim,
        agg_depth=meta.get("agg_depth", 12),
        num_heads=meta.get("num_heads", 16 if embed_dim == 1024 else embed_dim // 64),
        patch_embed=meta.get("patch_embed", "dinov2_vitl14_reg"),
        pretrained_patch_embed=False,
        enable_camera=True, enable_depth=True, enable_point=True, enable_track=False,
    ).to(device)


def load_student(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location="cpu")
    sd = ck.get("model", ck)
    meta = ck.get("meta", {})
    model = build_model(meta, device)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [k for k in missing if not k.startswith("track_head.")]
    unexpected = [k for k in unexpected if not k.startswith("track_head.")]
    if missing:
        print(f"  [警告] missing: {missing[:4]}")
    if unexpected:
        print(f"  [警告] unexpected: {unexpected[:4]}")
    if not missing and not unexpected:
        print("  [載入 OK] 完全匹配")
    model.eval()
    return model


def rotation_error_deg(R_pred, R_tgt):
    R = R_pred @ R_tgt.T
    c = (np.trace(R) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def _load_scene_images(scene, device, dtype, ids):
    meta = json.load(open(scene / "meta.json", encoding="utf-8"))
    imgs = []
    for i in ids:
        p = meta["image_paths"][i]
        im = cv2.cvtColor(cv2.imread(str(scene / "images" / p)), cv2.COLOR_BGR2RGB)
        imgs.append(torch.from_numpy(im.astype(np.float32) / 255.0).permute(2, 0, 1))
    images = torch.stack(imgs).unsqueeze(0).to(device=device, dtype=dtype)  # [1,S,3,H,W]
    return images, tuple(images.shape[-2:])


@torch.no_grad()
def eval_scene(model, scene, device, dtype, max_frames=12):
    meta = json.load(open(scene / "meta.json", encoding="utf-8"))
    n = meta["num_images"]
    # 抽 max_frames 幀（含 frame0，等距）避免大模型整段 50 幀 OOM
    ids = sorted(set(np.linspace(0, n - 1, min(max_frames, n)).astype(int).tolist()))

    images, hw = _load_scene_images(scene, device, dtype, ids)
    out = model(images)
    pred_pose = out["pose_enc"][0].float().cpu()        # [k,9]
    pred_depth = out["depth"][0].squeeze(-1).float().cpu().numpy()  # [k,H,W]

    tgt_pose = torch.from_numpy(np.load(scene / "pose_enc.npy")[ids]).float()
    tgt_depth = np.load(scene / "depth.npy")[ids]
    mask = np.load(scene / "point_mask.npy")[ids].astype(bool)

    pe, _ = pose_encoding_to_extri_intri(pred_pose.unsqueeze(0), hw)
    te, _ = pose_encoding_to_extri_intri(tgt_pose.unsqueeze(0), hw)
    pe, te = pe[0].numpy(), te[0].numpy()
    rot = np.array([rotation_error_deg(pe[i, :3, :3], te[i, :3, :3]) for i in range(pe.shape[0])])

    valid = mask & (tgt_depth > 1e-6) & (pred_depth > 1e-6)
    if valid.sum() > 100:
        p, t = pred_depth[valid].astype(np.float64), tgt_depth[valid].astype(np.float64)
        abs_rel = float(np.mean(np.abs(p - t) / t))
        delta1 = float(np.mean(np.maximum(p / t, t / p) < 1.25))
    else:
        abs_rel, delta1 = float("nan"), float("nan")

    return {"mean_rot_err": float(rot.mean()), "rra_15": float(np.mean(rot < 15)),
            "abs_rel": abs_rel, "delta_1": delta1}


def quick_eval(model, data_root_split, num_scenes, device, dtype):
    scenes = sorted([d for d in Path(data_root_split).iterdir()
                     if d.is_dir() and (d / "meta.json").exists()])[:num_scenes]
    rows = [eval_scene(model, s, device, dtype) for s in scenes]
    keys = rows[0].keys()
    return {k: float(np.nanmean([r[k] for r in rows])) for k in keys}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_root", required=True, help="test split 目錄")
    ap.add_argument("--num_scenes", type=int, default=10)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = load_student(args.ckpt, device)
    m = quick_eval(model, args.data_root, args.num_scenes, device, dtype)
    print("\n==== 評估 ====")
    for k, v in m.items():
        print(f"  {k:14s} {v:.4f}")


if __name__ == "__main__":
    main()

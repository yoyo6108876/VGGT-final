"""
distill2/visualize.py

視覺化結果（對應指導教授 P0-5）。輸出 PNG：
  1. rotation_histogram.png  ── 每幀旋轉誤差分布（可疊 collapsed 對照）
  2. camera_trajectory.png   ── 相機軌跡俯視圖：teacher vs student（可加 collapsed）
  3. depth_comparison.png    ── teacher vs student 深度圖（幾幀）
  4. pointcloud.png          ── 由深度反投影的 3D 點雲（teacher vs student）
  5. success_failure.png     ── 最好 / 最差場景的軌跡對照

圖上用英文標籤（避免 matplotlib 缺中文字型）。

用法：
  python visualize.py --ckpt runs/distill_b1/best.pt \
      --data_root D:/vggt-1/vggt-distillation/teacher_dataset_diverse/test \
      --out_dir viz --num_scenes 12
  # 加塌縮模型做前後對比（例如某個 collapsed 的 run）：
  python visualize.py --ckpt runs/distill_b1/best.pt \
      --collapsed_ckpt runs/distill2_s2/best.pt --data_root .../test
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eval import load_student, rotation_error_deg, _load_scene_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


def cam_centers(extri):
    """extri [S,3,4] → 相機中心 [S,3]（C = -R^T t）。"""
    C = []
    for i in range(extri.shape[0]):
        R, t = extri[i, :3, :3], extri[i, :3, 3]
        C.append(-R.T @ t)
    return np.stack(C)


def norm_shape(C):
    """置中並縮到單位 RMS 半徑，方便比較「形狀」（軌跡 vs 塌縮）。"""
    C = C - C.mean(0, keepdims=True)
    r = np.sqrt((C ** 2).sum(1).mean()) + 1e-9
    return C / r


@torch.no_grad()
def infer(model, scene, ids, device, dtype):
    images, hw = _load_scene_images(scene, device, dtype, ids)
    out = model(images)
    pose = out["pose_enc"][0].float().cpu()
    depth = out["depth"][0].squeeze(-1).float().cpu().numpy()
    extri, intri = pose_encoding_to_extri_intri(pose.unsqueeze(0), hw)
    return extri[0].numpy(), intri[0].numpy(), depth, hw


def frame_ids(scene, k):
    meta = json.load(open(scene / "meta.json", encoding="utf-8"))
    n = meta["num_images"]
    return sorted(set(np.linspace(0, n - 1, min(k, n)).astype(int).tolist()))


def scenes_of(data_root):
    return sorted([d for d in Path(data_root).iterdir() if d.is_dir() and (d / "meta.json").exists()])


# ────────────────────────────────────────────────
def fig_histogram(models, scenes, device, dtype, out, num_scenes):
    errs = {name: [] for name in models}
    for sc in scenes[:num_scenes]:
        ids = frame_ids(sc, 12)
        tgt = torch.from_numpy(np.load(sc / "pose_enc.npy")[ids]).float()
        te, _ = pose_encoding_to_extri_intri(tgt.unsqueeze(0), (518, 518)); te = te[0].numpy()
        for name, m in models.items():
            pe, _, _, _ = infer(m, sc, ids, device, dtype)
            for i in range(pe.shape[0]):
                errs[name].append(rotation_error_deg(pe[i, :3, :3], te[i, :3, :3]))
    plt.figure(figsize=(7, 4.2))
    colors = {"Path B (ours)": "#2E8B57", "collapsed": "#C0392B"}
    for name, e in errs.items():
        e = np.array(e)
        plt.hist(e, bins=np.linspace(0, 180, 37), alpha=0.6,
                 label=f"{name}  (mean {e.mean():.1f}°)", color=colors.get(name, None))
    plt.axvline(15, ls="--", c="gray", lw=1, label="RRA@15 threshold")
    plt.xlabel("Per-frame rotation error (deg)"); plt.ylabel("count")
    plt.title("Rotation error distribution"); plt.legend(); plt.tight_layout()
    plt.savefig(out / "rotation_histogram.png", dpi=150); plt.close()
    print("  ✓ rotation_histogram.png")


def fig_trajectory(models, scene, device, dtype, out):
    ids = frame_ids(scene, 20)
    tgt = torch.from_numpy(np.load(scene / "pose_enc.npy")[ids]).float()
    te, _ = pose_encoding_to_extri_intri(tgt.unsqueeze(0), (518, 518))
    Ct = norm_shape(cam_centers(te[0].numpy()))
    plt.figure(figsize=(6.2, 6))
    plt.plot(Ct[:, 0], Ct[:, 2], "-o", ms=4, c="#1f77b4", label="Teacher (target)")
    plt.scatter([Ct[0, 0]], [Ct[0, 2]], c="#1f77b4", s=120, marker="*", zorder=5)
    styles = {"Path B (ours)": ("#2E8B57", "-s"), "collapsed": ("#C0392B", "-^")}
    for name, m in models.items():
        pe, _, _, _ = infer(m, scene, ids, device, dtype)
        C = norm_shape(cam_centers(pe))
        c, st = styles.get(name, ("#888", "-x"))
        plt.plot(C[:, 0], C[:, 2], st, ms=4, c=c, alpha=0.85, label=name)
    plt.gca().set_aspect("equal"); plt.xlabel("X"); plt.ylabel("Z (top-down)")
    plt.title(f"Camera trajectory — {scene.name[:24]}\n(collapse = clustered, good = orbit)")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(out / "camera_trajectory.png", dpi=150); plt.close()
    print("  ✓ camera_trajectory.png")


def fig_depth(model, scene, device, dtype, out):
    ids = frame_ids(scene, 3)
    _, _, sd, _ = infer(model, scene, ids, device, dtype)
    td = np.load(scene / "depth.npy")[ids]
    n = len(ids)
    fig, ax = plt.subplots(n, 3, figsize=(9, 3 * n))
    ax = np.atleast_2d(ax)
    for r in range(n):
        vmax = np.percentile(td[r][td[r] > 0], 95) if (td[r] > 0).any() else 1
        ax[r, 0].imshow(td[r], cmap="turbo", vmin=0, vmax=vmax); ax[r, 0].set_title(f"Teacher depth (f{ids[r]})")
        ax[r, 1].imshow(sd[r], cmap="turbo", vmin=0, vmax=vmax); ax[r, 1].set_title("Student depth")
        d = np.abs(sd[r] - td[r]); ax[r, 2].imshow(d, cmap="magma"); ax[r, 2].set_title("|diff|")
        for c in range(3):
            ax[r, c].axis("off")
    plt.suptitle(f"Depth: Teacher vs Student — {scene.name[:24]}"); plt.tight_layout()
    plt.savefig(out / "depth_comparison.png", dpi=150); plt.close()
    print("  ✓ depth_comparison.png")


def unproject(depth, extri, intri, stride=8):
    H, W = depth.shape
    ys, xs = np.mgrid[0:H:stride, 0:W:stride]
    d = depth[::stride, ::stride]
    m = d > 1e-6
    xs, ys, d = xs[m], ys[m], d[m]
    fx, fy, cx, cy = intri[0, 0], intri[1, 1], intri[0, 2], intri[1, 2]
    cam = np.stack([(xs - cx) / fx * d, (ys - cy) / fy * d, d], 1)
    R, t = extri[:3, :3], extri[:3, 3]
    world = (cam - t) @ R
    return world


def fig_pointcloud(model, scene, device, dtype, out):
    ids = frame_ids(scene, 5)
    se, si, sd, _ = infer(model, scene, ids, device, dtype)
    tgt = torch.from_numpy(np.load(scene / "pose_enc.npy")[ids]).float()
    te, ti = pose_encoding_to_extri_intri(tgt.unsqueeze(0), (518, 518))
    te, ti = te[0].numpy(), ti[0].numpy()
    td = np.load(scene / "depth.npy")[ids]
    tw = np.concatenate([unproject(td[i], te[i], ti[i]) for i in range(len(ids))])
    sw = np.concatenate([unproject(sd[i], se[i], si[i]) for i in range(len(ids))])
    fig = plt.figure(figsize=(11, 5.5))
    for j, (W_, title) in enumerate([(tw, "Teacher point cloud"), (sw, "Student point cloud")]):
        axp = fig.add_subplot(1, 2, j + 1, projection="3d")
        idx = np.random.choice(len(W_), min(6000, len(W_)), replace=False)
        axp.scatter(W_[idx, 0], W_[idx, 2], -W_[idx, 1], s=1, c=-W_[idx, 1], cmap="viridis")
        axp.set_title(title); axp.set_xticks([]); axp.set_yticks([]); axp.set_zticks([])
    plt.suptitle(f"Point cloud — {scene.name[:24]}"); plt.tight_layout()
    plt.savefig(out / "pointcloud.png", dpi=150); plt.close()
    print("  ✓ pointcloud.png")


def fig_success_failure(model, scenes, device, dtype, out, num_scenes):
    per = []
    for sc in scenes[:num_scenes]:
        ids = frame_ids(sc, 12)
        tgt = torch.from_numpy(np.load(sc / "pose_enc.npy")[ids]).float()
        te, _ = pose_encoding_to_extri_intri(tgt.unsqueeze(0), (518, 518)); te = te[0].numpy()
        pe, _, _, _ = infer(model, sc, ids, device, dtype)
        errs = [rotation_error_deg(pe[i, :3, :3], te[i, :3, :3]) for i in range(pe.shape[0])]
        per.append((np.mean(errs), sc, pe, te))
    per.sort(key=lambda x: x[0])
    best, worst = per[0], per[-1]
    fig, ax = plt.subplots(1, 2, figsize=(11, 5.4))
    for a, (err, sc, pe, te), tag in [(ax[0], best, "Best"), (ax[1], worst, "Worst")]:
        Ct = norm_shape(cam_centers(te)); Cs = norm_shape(cam_centers(pe))
        a.plot(Ct[:, 0], Ct[:, 2], "-o", ms=4, c="#1f77b4", label="Teacher")
        a.plot(Cs[:, 0], Cs[:, 2], "-s", ms=4, c="#2E8B57", label="Student")
        a.set_aspect("equal"); a.grid(alpha=0.3); a.legend()
        a.set_title(f"{tag}: {sc.name[:20]}\nmean rot err {err:.1f}°")
    plt.suptitle("Success vs Failure cases (camera trajectory)"); plt.tight_layout()
    plt.savefig(out / "success_failure.png", dpi=150); plt.close()
    print(f"  ✓ success_failure.png（best {best[0]:.1f}° / worst {worst[0]:.1f}°）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="student（例如 Path B best.pt）")
    ap.add_argument("--collapsed_ckpt", default=None, help="塌縮模型（做前後對比，選填）")
    ap.add_argument("--data_root", required=True, help="test split 目錄")
    ap.add_argument("--scene", default=None, help="指定示範場景名稱（預設取第一個）")
    ap.add_argument("--num_scenes", type=int, default=12, help="histogram / success-failure 用幾場景")
    ap.add_argument("--out_dir", default="viz")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    print("載入 student ...")
    student = load_student(args.ckpt, device)
    models = {"Path B (ours)": student}
    if args.collapsed_ckpt:
        print("載入 collapsed 對照 ...")
        models["collapsed"] = load_student(args.collapsed_ckpt, device)

    scenes = scenes_of(args.data_root)
    demo = next((s for s in scenes if s.name == args.scene), scenes[0]) if args.scene else scenes[0]
    print(f"示範場景：{demo.name}\n")

    for fn in [
        lambda: fig_histogram(models, scenes, device, dtype, out, args.num_scenes),
        lambda: fig_trajectory(models, demo, device, dtype, out),
        lambda: fig_depth(student, demo, device, dtype, out),
        lambda: fig_pointcloud(student, demo, device, dtype, out),
        lambda: fig_success_failure(student, scenes, device, dtype, out, args.num_scenes),
    ]:
        try:
            fn()
        except Exception as e:
            print(f"  [略過一張圖] {type(e).__name__}: {e}")

    print(f"\n完成，圖存於：{out}/")


if __name__ == "__main__":
    main()

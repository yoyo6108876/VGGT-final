r"""
distill2/eval_gt.py

嚴謹的相對位姿評估（對應指導教授 P0-1）。
用「相對位姿」指標（對全域座標系不變，是多視角位姿的標準評估）：
  RRA@τ  相對旋轉誤差 < τ 的配對比例
  RTA@τ  相對平移方向誤差 < τ 的配對比例
  AUC@30 relative-rotation 準確率曲線在 1~30° 的面積（mAA）

參考來源（--ref）：
  teacher（預設）：以 teacher（原版 VGGT）的 pose 當參考 → 量 student 與原版 VGGT 的相對位姿一致性。
  co3d          ：載入 CO3D ground-truth pose，把 student 與 teacher 都對 GT 評估
                  → 直接檢驗「student 接近 teacher ≠ 接近 GT」。需 --co3d_root 指向 CO3D 標註。

用法：
  python eval_gt.py --ckpt runs/distill_b1/best.pt --data_root .../test --num_scenes 20
  python eval_gt.py --ckpt runs/distill_b1/best.pt --data_root .../test --ref co3d --co3d_root D:/co3d_data
"""

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import torch

from eval import load_student, _load_scene_images
from quantize import load_any   # 一般/切片 ckpt 皆可載入
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


# ────────── 相對位姿指標 ──────────
def _rot_angle(Ra, Rb):
    R = Ra @ Rb.T
    c = np.clip((np.trace(R) - 1) / 2, -1, 1)
    return np.degrees(np.arccos(c))


def _vec_angle(a, b):
    a = a / (np.linalg.norm(a) + 1e-9); b = b / (np.linalg.norm(b) + 1e-9)
    return np.degrees(np.arccos(np.clip(a @ b, -1, 1)))


def rel_pose_errors(pred, ref):
    """pred/ref: [S,3,4] OpenCV world2cam。回傳每個配對的 (rot_err, trans_err) 度。"""
    S = pred.shape[0]
    re, te = [], []
    for i in range(S):
        Ri_p, Rj = pred[i, :3, :3], None
        Ci_p = -pred[i, :3, :3].T @ pred[i, :3, 3]
        Ci_r = -ref[i, :3, :3].T @ ref[i, :3, 3]
        for j in range(i + 1, S):
            Rrel_p = pred[j, :3, :3] @ pred[i, :3, :3].T
            Rrel_r = ref[j, :3, :3] @ ref[i, :3, :3].T
            re.append(_rot_angle(Rrel_p, Rrel_r))
            Cj_p = -pred[j, :3, :3].T @ pred[j, :3, 3]
            Cj_r = -ref[j, :3, :3].T @ ref[j, :3, 3]
            tp = pred[i, :3, :3] @ (Cj_p - Ci_p)   # cam i 座標系下，j 相對 i 的方向
            tr = ref[i, :3, :3] @ (Cj_r - Ci_r)
            te.append(_vec_angle(tp, tr))
    return np.array(re), np.array(te)


def summarize(re, te):
    def acc(e, t): return float((e < t).mean()) if len(e) else float("nan")
    auc = float(np.mean([(re < t).mean() for t in range(1, 31)])) if len(re) else float("nan")
    return {"RRA@15": acc(re, 15), "RRA@30": acc(re, 30),
            "RTA@15": acc(te, 15), "RTA@30": acc(te, 30),
            "AUC@30": auc, "n_pairs": len(re)}


# ────────── CO3D GT 載入 ──────────
def pt3d_to_opencv(R, T, convention="flip_xy"):
    """PyTorch3D (CO3D) 相機 → OpenCV world2cam [R|t]。
    PyTorch3D: X_view = X_world @ R + T（row-vector）；轉成 column-vector 需轉置。
    再翻到 OpenCV 軸向（+X 右、+Y 下、+Z 前）。convention 可切換以便對照。"""
    R = np.asarray(R, np.float64); T = np.asarray(T, np.float64)
    Rc = R.T          # column-vector world2cam
    tc = T.copy()
    if convention == "flip_xy":       # 最常見：翻 X、Y
        F = np.diag([-1.0, -1.0, 1.0])
        Rc = F @ Rc; tc = F @ tc
    elif convention == "flip_yz":
        F = np.diag([1.0, -1.0, -1.0])
        Rc = F @ Rc; tc = F @ tc
    elif convention == "none":
        pass
    return np.concatenate([Rc, tc[:, None]], axis=1)   # [3,4]


def load_co3d_gt(co3d_root, scene_name, source_names, convention):
    """依 scene_name（category__seqid）與原始影像檔名，載入 CO3D GT 外參 [S,3,4]。"""
    if "__" not in scene_name:
        return None
    category, seqid = scene_name.split("__", 1)
    ann = Path(co3d_root) / category / "frame_annotations.jgz"
    if not ann.exists():
        return None
    with gzip.open(ann, "rt", encoding="utf-8") as f:
        data = json.load(f)
    want = set(source_names)
    vp_by_name = {}
    for a in data:
        if a.get("sequence_name") != seqid:
            continue
        name = Path(a["image"]["path"]).name
        if name in want:
            vp_by_name[name] = a["viewpoint"]
    extri = []
    for name in source_names:
        vp = vp_by_name.get(name)
        if vp is None:
            return None
        extri.append(pt3d_to_opencv(vp["R"], vp["T"], convention))
    return np.stack(extri)


# ────────── 每場景取外參 ──────────
def frame_ids(meta, k, rng=None):
    """預設等距取幀；給 rng 時改為隨機取幀（貼近文獻協定：隨機 10 幀）。"""
    n = meta["num_images"]
    if rng is not None:
        return sorted(rng.choice(n, size=min(k, n), replace=False).tolist())
    return sorted(set(np.linspace(0, n - 1, min(k, n)).astype(int).tolist()))


@torch.no_grad()
def student_extri(model, scene, ids, device, dtype):
    imgs, hw = _load_scene_images(scene, device, dtype, ids)
    pose = model(imgs)["pose_enc"][0].float().cpu()
    e, _ = pose_encoding_to_extri_intri(pose.unsqueeze(0), hw)
    return e[0].numpy()


def teacher_extri(scene, ids):
    pose = torch.from_numpy(np.load(scene / "pose_enc.npy")[ids]).float()
    e, _ = pose_encoding_to_extri_intri(pose.unsqueeze(0), (518, 518))
    return e[0].numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_root", required=True, help="test split 目錄")
    ap.add_argument("--num_scenes", type=int, default=20)
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--ref", choices=["teacher", "co3d"], default="teacher")
    ap.add_argument("--co3d_root", default=None, help="CO3D 資料根目錄（含各類別 frame_annotations.jgz）")
    ap.add_argument("--co3d_convention", choices=["flip_xy", "flip_yz", "none"], default="flip_xy")
    ap.add_argument("--random_frames", type=int, default=-1,
                    help="≥0 時改為隨機取幀（值為 seed），貼近文獻協定的『隨機 10 幀』")
    ap.add_argument("--out_prefix", default="eval_gt_report")
    args = ap.parse_args()

    if args.ref == "co3d" and not args.co3d_root:
        raise SystemExit("--ref co3d 需要 --co3d_root")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    student = load_any(args.ckpt, device)

    scenes = sorted([d for d in Path(args.data_root).iterdir() if d.is_dir() and (d / "meta.json").exists()])[:args.num_scenes]
    # methods → 累積各 pair 誤差
    methods = ["student"] + (["teacher"] if args.ref == "co3d" else [])
    RE = {m: [] for m in methods}; TE = {m: [] for m in methods}
    n_used, n_skip = 0, 0

    rng = np.random.default_rng(args.random_frames) if args.random_frames >= 0 else None
    for sc in scenes:
        meta = json.load(open(sc / "meta.json", encoding="utf-8"))
        ids = frame_ids(meta, args.frames, rng)
        src = [meta["source_images"][i] for i in ids]

        if args.ref == "teacher":
            ref = teacher_extri(sc, ids)
        else:
            ref = load_co3d_gt(args.co3d_root, meta["scene_name"], src, args.co3d_convention)
            if ref is None:
                n_skip += 1; continue

        se = student_extri(student, sc, ids, device, dtype)
        re, te = rel_pose_errors(se, ref); RE["student"].append(re); TE["student"].append(te)
        if "teacher" in methods:
            tex = teacher_extri(sc, ids)
            re2, te2 = rel_pose_errors(tex, ref); RE["teacher"].append(re2); TE["teacher"].append(te2)
        n_used += 1

    if n_used == 0:
        raise SystemExit(f"沒有可評估的場景（skip={n_skip}）。若用 co3d，請確認 co3d_root 有 frame_annotations.jgz 且序列名對得上。")

    print(f"\n參考來源：{args.ref}"
          + (f"（CO3D，convention={args.co3d_convention}）" if args.ref == "co3d" else "（teacher / 原版 VGGT）")
          + f"｜評估場景 {n_used}（skip {n_skip}）｜每場景 {args.frames} 幀\n")

    results = {}
    for m in methods:
        re = np.concatenate(RE[m]); te = np.concatenate(TE[m])
        results[m] = summarize(re, te)

    cols = ["RRA@15", "RRA@30", "RTA@15", "RTA@30", "AUC@30"]
    print("{:<10} {:>8} {:>8} {:>8} {:>8} {:>8}".format("method", *cols))
    for m in methods:
        r = results[m]
        print("{:<10} {:>8.3f} {:>8.3f} {:>8.3f} {:>8.3f} {:>8.3f}".format(m, *[r[c] for c in cols]))
    if args.ref == "co3d":
        print("\n（student 與 teacher 都對 GT 評估：若兩者接近，代表 student 忠實蒸餾了 teacher；\n"
              " teacher 對 GT 的誤差就是 student 精度的上限 —— 直接回答『接近 teacher ≠ 接近 GT』。）")
        if results.get("teacher", {}).get("RRA@15", 0) < 0.3:
            print("⚠ teacher 對 GT 的 RRA@15 偏低 → CO3D 座標轉換可能不對，試 --co3d_convention flip_yz/none 對照。")

    # 輸出
    md = [f"# 相對位姿評估（P0-1）\n",
          f"- 參考：{args.ref}｜場景 {n_used}｜每場景 {args.frames} 幀\n",
          "| method | RRA@15 | RRA@30 | RTA@15 | RTA@30 | AUC@30 |",
          "|---|---|---|---|---|---|"]
    for m in methods:
        r = results[m]
        md.append("| {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} |".format(m, *[r[c] for c in cols]))
    Path(args.out_prefix + ".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\n已輸出：{args.out_prefix}.md")


if __name__ == "__main__":
    main()

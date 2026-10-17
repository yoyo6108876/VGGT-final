"""
distill2/slicegpt_probe.py

SliceGPT 可行性探針（光譜截斷診斷）。

SliceGPT（Ashkboos et al., ICLR 2024）的核心：對殘差流做 activation-PCA
正交旋轉，再切掉尾端維度——切片後的模型「只能看到前 k 個主成分」。
在投入完整權重手術（LN→RMS 摺疊、逐層 Q、head 介面摺疊）之前，
本腳本用 forward hook 直接模擬這件事：

    每個 aggregator block 輸出 y  →  μ + (y−μ) Q_k Q_kᵀ

即把殘差流限制在校準資料 PCA 的前 k 維子空間，掃 k 看 pose 何時崩。
  - k=1024 應等於 baseline（sanity check）
  - 若 k≈512 仍存活 → SliceGPT 式寬度壓縮有戲，值得做完整版
  - 若 k≈896 就崩   → pose 路徑真的需要全寬度，SliceGPT 也救不了（省下數天工程）

用法（在 distill2/ 下）：
  python slicegpt_probe.py --ckpt runs/distill_b1/best.pt \
      --data_root D:\vggt-1\vggt-distillation\teacher_dataset_diverse \
      --ks 1024,896,768,640,512,384,256
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from eval import load_student, quick_eval


# ──────────────────────────────────────────────
# 校準：累積每個 block 輸出的均值 / 共變異數 → PCA 基底
# ──────────────────────────────────────────────

class CovAccumulator:
    def __init__(self, dim, device):
        self.n = 0
        self.s = torch.zeros(dim, dtype=torch.float64, device=device)
        self.c = torch.zeros(dim, dim, dtype=torch.float64, device=device)

    @torch.no_grad()
    def update(self, x):            # x: [..., C]
        x = x.reshape(-1, x.shape[-1]).double()
        self.n += x.shape[0]
        self.s += x.sum(0)
        self.c += x.T @ x

    def pca(self):
        mu = self.s / self.n
        cov = self.c / self.n - torch.outer(mu, mu)
        evals, evecs = torch.linalg.eigh(cov)          # 升冪
        order = torch.argsort(evals, descending=True)
        return mu.float(), evecs[:, order].float(), evals[order].float()


def iter_blocks(model):
    """殘差流上的所有 hook 位置：frame/global blocks 交替（依 forward 順序）。"""
    agg = model.aggregator
    for i in range(agg.depth):
        yield f"frame[{i}]", agg.frame_blocks[i]
        yield f"global[{i}]", agg.global_blocks[i]


@torch.no_grad()
def calibrate(model, calib_root, num_scenes, num_frames, device, dtype):
    """在 train split 上跑前向，收集每個 block 輸出的 PCA。"""
    names = [n for n, _ in iter_blocks(model)]
    accs = {n: CovAccumulator(model.embed_dim, device) for n in names}

    handles = []
    for name, blk in iter_blocks(model):
        def make_hook(n):
            def hook(module, inputs, output):
                accs[n].update(output)
            return hook
        handles.append(blk.register_forward_hook(make_hook(name)))

    scenes = sorted([d for d in Path(calib_root).iterdir()
                     if d.is_dir() and (d / "meta.json").exists()])[:num_scenes]
    from eval import _load_scene_images
    for sc in scenes:
        meta = json.load(open(sc / "meta.json", encoding="utf-8"))
        n = meta["num_images"]
        ids = sorted(set(np.linspace(0, n - 1, min(num_frames, n)).astype(int).tolist()))
        images, _ = _load_scene_images(sc, device, dtype, ids)
        model(images)
    for h in handles:
        h.remove()

    bases = {}
    for n in names:
        mu, Q, evals = accs[n].pca()
        bases[n] = (mu, Q, evals)
    return bases


# ──────────────────────────────────────────────
# 截斷 eval：把每個 block 輸出投影到前 k 個主成分
# ──────────────────────────────────────────────

def install_truncation(model, bases, k):
    handles = []
    for name, blk in iter_blocks(model):
        mu, Q, _ = bases[name]
        P = (Q[:, :k] @ Q[:, :k].T).contiguous()       # [C, C] fp32

        def make_hook(mu=mu, P=P):
            def hook(module, inputs, output):
                y = output.float()
                y = (y - mu) @ P + mu
                return y.to(output.dtype)
            return hook
        handles.append(blk.register_forward_hook(make_hook()))
    return handles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=r"runs\distill_b1\best.pt")
    ap.add_argument("--data_root", default=r"D:\vggt-1\vggt-distillation\teacher_dataset_diverse")
    ap.add_argument("--calib_scenes", type=int, default=8)
    ap.add_argument("--calib_frames", type=int, default=6)
    ap.add_argument("--eval_scenes", type=int, default=12)
    ap.add_argument("--ks", default="1024,896,768,640,512,384,256")
    ap.add_argument("--out", default=r"reports\slicegpt_probe.md")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    ks = [int(x) for x in args.ks.split(",")]

    model = load_student(args.ckpt, device)

    print(f"校準（train split，{args.calib_scenes} scenes × {args.calib_frames} frames）...", flush=True)
    bases = calibrate(model, str(Path(args.data_root) / "train"),
                      args.calib_scenes, args.calib_frames, device, dtype)

    # 各 site 的能量集中度（前 k 個特徵值佔比）供報告用
    energy = {}
    for name, (_, _, ev) in bases.items():
        tot = ev.sum()
        energy[name] = {k: float(ev[:k].sum() / tot) for k in ks}

    test_root = str(Path(args.data_root) / "test")
    rows = []
    baseline = quick_eval(model, test_root, args.eval_scenes, device, dtype)
    print(f"baseline（無截斷）: rot_err={baseline['mean_rot_err']:.2f}° "
          f"rra15={baseline['rra_15']:.3f} abs_rel={baseline['abs_rel']:.4f}", flush=True)
    rows.append({"k": "無截斷", **baseline, "energy": 1.0})

    for k in ks:
        handles = install_truncation(model, bases, k)
        m = quick_eval(model, test_root, args.eval_scenes, device, dtype)
        for h in handles:
            h.remove()
        mean_energy = float(np.mean([energy[n][k] for n in energy]))
        rows.append({"k": k, **m, "energy": mean_energy})
        print(f"k={k:4d}: rot_err={m['mean_rot_err']:.2f}° rra15={m['rra_15']:.3f} "
              f"abs_rel={m['abs_rel']:.4f}  (平均 PCA 能量 {mean_energy*100:.2f}%)", flush=True)

    md = ["# SliceGPT 可行性探針（光譜截斷診斷）\n",
          f"模型：`{args.ckpt}`（Path B）。把每個 aggregator block 輸出投影到",
          "activation-PCA 前 k 維子空間（= SliceGPT 切片後模型所能保留的資訊上限），",
          f"校準 {args.calib_scenes} 個 train scenes，eval {args.eval_scenes} 個 test scenes。\n",
          "| k（保留維度） | 平均 PCA 能量 | rot_err | RRA@15 | abs_rel | delta_1 |",
          "|---|---|---|---|---|---|"]
    for r in rows:
        e = f"{r['energy']*100:.2f}%" if isinstance(r["energy"], float) else "—"
        md.append(f"| {r['k']} | {e} | {r['mean_rot_err']:.2f}° | {r['rra_15']:.3f} "
                  f"| {r['abs_rel']:.4f} | {r['delta_1']:.3f} |")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"已輸出 {out}")


if __name__ == "__main__":
    main()

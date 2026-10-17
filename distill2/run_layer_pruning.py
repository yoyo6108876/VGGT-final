"""
distill2/run_layer_pruning.py

層剪枝（layer pruning）實驗 orchestrator。
保持寬度 1024 不動（縮寬已證明會塌），只砍 aggregator 層數：
teacher 24 層 → student depth ∈ {12(基準), 8, 6, 4}，
複製策略沿用 Path B（stride 取層：j = i * (24 // depth)），
其餘配方完全同消融實驗（--copy all、凍結 patch embed、同訓練超參），
藉此隔離「層數」單一變因。

用法：
  python run_layer_pruning.py --stage all --epochs 12 --img_per_seq 2 --throttle 0.05
  python run_layer_pruning.py --stage collect            # 只把已完成的 run 收成表
  python run_layer_pruning.py --stage all --depths 8,6
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import torch

PY = sys.executable
HERE = Path(__file__).parent

TEACHER_PARAMS_M = 1256.0   # VGGT-1B（見 reports/efficiency_report.md）
BASELINE_DEPTH = 12         # Path B 基準（runs/abl/full）


def sh(cmd):
    print("＄ " + " ".join(str(c) for c in cmd), flush=True)
    # PYTHONUTF8=1：子行程輸出被重導向時，Windows 會退回 cp950，
    # 印 ✓ 等字元會 UnicodeEncodeError 使訓練中途死掉
    env = {**os.environ, "PYTHONUTF8": "1"}
    subprocess.run([str(c) for c in cmd], check=True, env=env)


def params_m(ckpt_path):
    ck = torch.load(ckpt_path, map_location="cpu")
    sd = ck.get("model", ck)
    return sum(v.numel() for v in sd.values()) / 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["build", "train", "collect", "all"], default="all")
    ap.add_argument("--depths", default="8,6,4", help="逗號分隔的 student 層數（24 的因數較佳）")
    ap.add_argument("--data_root", default=r"D:\vggt-1\vggt-distillation\teacher_dataset_diverse")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--steps_per_epoch", type=int, default=400)
    ap.add_argument("--img_per_seq", type=int, default=2)
    ap.add_argument("--eval_scenes", type=int, default=8)
    ap.add_argument("--throttle", type=float, default=0.0)
    ap.add_argument("--out_prefix", default="reports/layer_pruning_report")
    args = ap.parse_args()

    depths = [int(x) for x in args.depths.split(",") if x.strip()]
    abl = HERE / "abl"; abl.mkdir(exist_ok=True)
    runs = HERE / "runs" / "prune"; runs.mkdir(parents=True, exist_ok=True)

    if args.stage in ("build", "all"):
        for d in depths:
            init = abl / f"prune_d{d}_init.pt"
            print(f"\n===== BUILD depth={d} =====", flush=True)
            sh([PY, HERE / "build_student.py", "--out", init, "--copy", "all",
                "--student_depth", d])

    if args.stage in ("train", "all"):
        for d in depths:
            init = abl / f"prune_d{d}_init.pt"
            out = runs / f"d{d}"
            print(f"\n===== TRAIN depth={d} =====", flush=True)
            sh([PY, HERE / "train.py", "--student_init", init, "--data_root", args.data_root,
                "--out_dir", out, "--epochs", args.epochs, "--steps_per_epoch", args.steps_per_epoch,
                "--img_per_seq", args.img_per_seq, "--eval_scenes", args.eval_scenes,
                "--eval_every", 3, "--freeze_patch_embed", 1, "--throttle", args.throttle])

    if args.stage in ("collect", "all"):
        rows = []
        # 基準：Path B depth=12（消融 full 組，同訓練配方）
        baseline = HERE / "runs" / "abl" / "full" / "best.pt"
        candidates = [(BASELINE_DEPTH, baseline, "Path B 基準")] + \
                     [(d, runs / f"d{d}" / "best.pt", "剪枝") for d in depths]
        for d, best, tag in candidates:
            r = {"depth": d, "tag": tag}
            if best.exists():
                ck = torch.load(best, map_location="cpu")
                e = ck.get("eval", {})
                p = params_m(best)
                r.update({"params_m": p, "ratio": TEACHER_PARAMS_M / p,
                          "rot_err": e.get("mean_rot_err"), "rra_15": e.get("rra_15"),
                          "abs_rel": e.get("abs_rel"), "delta_1": e.get("delta_1"),
                          "epoch": ck.get("epoch")})
            else:
                r.update({k: None for k in ["params_m", "ratio", "rot_err", "rra_15",
                                            "abs_rel", "delta_1", "epoch"]})
            rows.append(r)

        def f(v, s="", nd=2):
            return "—" if v is None else f"{v:.{nd}f}{s}"

        print("\n" + "=" * 100)
        print("層剪枝結果（寬度固定 1024；teacher 24 層 stride 取層 + Path B 配方）")
        print("=" * 100)
        print("{:<6} {:<12} {:>10} {:>8} {:>9} {:>8} {:>8} {:>8}".format(
            "depth", "tag", "params(M)", "壓縮×", "rot_err", "RRA@15", "abs_rel", "d1"))
        for r in rows:
            print("{:<6} {:<12} {:>10} {:>8} {:>9} {:>8} {:>8} {:>8}".format(
                r["depth"], r["tag"], f(r["params_m"], "", 0), f(r["ratio"], "×"),
                f(r["rot_err"], "°"), f(r["rra_15"], "", 3), f(r["abs_rel"], "", 3),
                f(r["delta_1"], "", 3)))
        print("=" * 100)

        md = ["# 層剪枝實驗（保持寬度 1024、只砍 aggregator 層數）\n",
              "Teacher 24 層。所有組合皆為 Path B 配方（--copy all + 凍結 patch embed），",
              f"stride 取層；訓練 {args.epochs} epochs × {args.steps_per_epoch} steps、"
              f"img_per_seq={args.img_per_seq}（與消融實驗同配方）。",
              f"Teacher = {TEACHER_PARAMS_M:.0f}M 參數。\n",
              "| depth | 說明 | params (M) | 壓縮× | rot_err | RRA@15 | abs_rel | delta_1 |",
              "|---|---|---|---|---|---|---|---|"]
        for r in rows:
            md.append("| {} | {} | {} | {} | {} | {} | {} | {} |".format(
                r["depth"], r["tag"], f(r["params_m"], "", 0), f(r["ratio"], "×"),
                f(r["rot_err"], "°"), f(r["rra_15"], "", 3), f(r["abs_rel"], "", 3),
                f(r["delta_1"], "", 3)))
        out_md = HERE / f"{args.out_prefix}.md"
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text("\n".join(md) + "\n", encoding="utf-8")
        import csv
        with open(HERE / f"{args.out_prefix}.csv", "w", newline="", encoding="utf-8") as fp:
            w = csv.writer(fp)
            w.writerow(["depth", "tag", "params_m", "ratio", "rot_err", "rra_15",
                        "abs_rel", "delta_1", "epoch"])
            for r in rows:
                w.writerow([r["depth"], r["tag"], r["params_m"], r["ratio"], r["rot_err"],
                            r["rra_15"], r["abs_rel"], r["delta_1"], r["epoch"]])
        print(f"已輸出：{args.out_prefix}.md、{args.out_prefix}.csv")


if __name__ == "__main__":
    main()

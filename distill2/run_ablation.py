"""
distill2/run_ablation.py

消融實驗 orchestrator（對應指導教授 P0-4）：逐一拆解「teacher-init 哪個元件在起作用」。
所有組合都是同寬度（1024，~842M）—— 差別只在「初始化」，藉此隔離各元件的貢獻。

組合：
  random        : 不複製（patch embed 用 hub 預訓練，aggregator/camera head 隨機）
  patch_only    : 只複製 patch embed
  agg_only      : 只複製 aggregator
  head_only     : 只複製 camera head
  full          : 複製全部（＝ Path B；patch embed 凍結）
  full_ft_pe    : 複製全部，但 patch embed 微調（frozen vs fine-tune 對照）

用法：
  python run_ablation.py --stage all --epochs 15 --img_per_seq 2 --throttle 0.05
  python run_ablation.py --stage collect                 # 只把已完成的 run 收成表
  python run_ablation.py --stage all --only head_only,full
"""

import argparse
import subprocess
import sys
from pathlib import Path

import torch

PY = sys.executable
HERE = Path(__file__).parent

# (name, copy 字串, freeze_patch_embed)
CONFIGS = [
    ("random",     "none",                              1),
    ("patch_only", "patch_embed",                       1),
    ("agg_only",   "aggregator",                        1),
    ("head_only",  "camera_head",                       1),
    ("full",       "patch_embed,aggregator,camera_head", 1),  # = Path B
    ("full_ft_pe", "patch_embed,aggregator,camera_head", 0),  # patch embed 微調
]


def sh(cmd):
    print("＄ " + " ".join(str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["build", "train", "collect", "all"], default="all")
    ap.add_argument("--data_root", default=r"D:\vggt-1\vggt-distillation\teacher_dataset_diverse")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--steps_per_epoch", type=int, default=400)
    ap.add_argument("--img_per_seq", type=int, default=2)
    ap.add_argument("--eval_scenes", type=int, default=8)
    ap.add_argument("--throttle", type=float, default=0.0)
    ap.add_argument("--only", default="", help="逗號分隔，只跑部分組合")
    ap.add_argument("--out_prefix", default="ablation_report")
    args = ap.parse_args()

    only = set(x.strip() for x in args.only.split(",") if x.strip())
    configs = [c for c in CONFIGS if (not only or c[0] in only)]
    abl = HERE / "abl"; abl.mkdir(exist_ok=True)
    runs = HERE / "runs" / "abl"; runs.mkdir(parents=True, exist_ok=True)

    if args.stage in ("build", "all"):
        for name, copy, _ in configs:
            init = abl / f"{name}_init.pt"
            print(f"\n===== BUILD {name} (copy={copy}) =====")
            sh([PY, HERE / "build_student.py", "--out", init, "--copy", copy])

    if args.stage in ("train", "all"):
        for name, copy, freeze in configs:
            init = abl / f"{name}_init.pt"
            out = runs / name
            print(f"\n===== TRAIN {name} =====")
            sh([PY, HERE / "train.py", "--student_init", init, "--data_root", args.data_root,
                "--out_dir", out, "--epochs", args.epochs, "--steps_per_epoch", args.steps_per_epoch,
                "--img_per_seq", args.img_per_seq, "--eval_scenes", args.eval_scenes,
                "--eval_every", 3, "--freeze_patch_embed", freeze, "--throttle", args.throttle])

    if args.stage in ("collect", "all"):
        rows = []
        for name, copy, freeze in configs:
            best = runs / name / "best.pt"
            r = {"name": name, "copy": copy, "freeze_pe": freeze}
            if best.exists():
                ck = torch.load(best, map_location="cpu")
                e = ck.get("eval", {})
                r.update({"rot_err": e.get("mean_rot_err"), "rra_15": e.get("rra_15"),
                          "abs_rel": e.get("abs_rel"), "delta_1": e.get("delta_1"),
                          "epoch": ck.get("epoch")})
            else:
                r.update({"rot_err": None, "rra_15": None, "abs_rel": None, "delta_1": None, "epoch": None})
            rows.append(r)

        def f(v, s="", nd=2):
            return "—" if v is None else f"{v:.{nd}f}{s}"

        print("\n" + "=" * 96)
        print("消融結果（同寬度 1024 ~842M；差別僅在初始化）")
        print("=" * 96)
        print("{:<12} {:<34} {:>7} {:>9} {:>8} {:>8} {:>8}".format("組合", "複製元件", "freeze", "rot_err", "RRA@15", "abs_rel", "d1"))
        for r in rows:
            print("{:<12} {:<34} {:>7} {:>9} {:>8} {:>8} {:>8}".format(
                r["name"], r["copy"], r["freeze_pe"], f(r["rot_err"], "°"), f(r["rra_15"], "", 3),
                f(r["abs_rel"], "", 3), f(r["delta_1"], "", 3)))
        print("=" * 96)

        md = ["# 消融實驗（Teacher-init 各元件的貢獻）\n",
              "所有組合皆為同寬度 1024（~842M）；差別只在「初始化」，藉此隔離各元件貢獻。\n",
              "| 組合 | 複製元件 | freeze PE | rot_err | RRA@15 | abs_rel | delta_1 |",
              "|---|---|---|---|---|---|---|"]
        for r in rows:
            md.append("| {} | {} | {} | {} | {} | {} | {} |".format(
                r["name"], r["copy"], r["freeze_pe"], f(r["rot_err"], "°"), f(r["rra_15"], "", 3),
                f(r["abs_rel"], "", 3), f(r["delta_1"], "", 3)))
        (HERE / f"{args.out_prefix}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
        import csv
        with open(HERE / f"{args.out_prefix}.csv", "w", newline="", encoding="utf-8") as fp:
            w = csv.writer(fp)
            w.writerow(["name", "copy", "freeze_pe", "rot_err", "rra_15", "abs_rel", "delta_1", "epoch"])
            for r in rows:
                w.writerow([r["name"], r["copy"], r["freeze_pe"], r["rot_err"], r["rra_15"],
                            r["abs_rel"], r["delta_1"], r["epoch"]])
        print(f"已輸出：{args.out_prefix}.md、{args.out_prefix}.csv")


if __name__ == "__main__":
    main()

"""
distill2/train.py

乾淨的蒸餾訓練迴圈（無 Hydra / DDP）：
  - 從 build_student.py 產出的 student_init.pt 載入（已繼承 teacher 權重）
  - 凍結 DINOv2 patch embed（省 VRAM、保住預訓練特徵）
  - AdamW + warmup/cosine + grad clip + AMP(bf16)
  - 定期 eval（mean_rot_err / abs_rel）+ 存 best

用法：
  python train.py --student_init student_init.pt \
      --data_root D:\vggt-1\vggt-distillation\teacher_dataset_diverse \
      --out_dir runs/distill_b1
"""

import argparse
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data import TeacherDataset
from loss import DistillLoss
from eval import build_model, quick_eval


def cosine_warmup(step, total, warmup, base_lr, min_lr=1e-7):
    if step < warmup:
        return base_lr * step / max(1, warmup)
    t = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * t))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student_init", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out_dir", default="runs/distill_b1")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--steps_per_epoch", type=int, default=500)
    ap.add_argument("--img_per_seq", type=int, default=4)
    ap.add_argument("--batch_scenes", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--warmup_frac", type=float, default=0.05)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--eval_every", type=int, default=2, help="每幾個 epoch eval 一次")
    ap.add_argument("--eval_scenes", type=int, default=8)
    ap.add_argument("--freeze_patch_embed", type=int, default=1)
    ap.add_argument("--throttle", type=float, default=0.0,
                    help="每步額外 sleep 秒數，降低 GPU 佔用避免螢幕黑頻（例 0.05~0.15）")
    ap.add_argument("--gt_pose", type=int, default=0,
                    help="1：pose 目標改用 CO3D GT 外參（需先跑 make_gt_labels.py）")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    # ── 模型（繼承 teacher 權重）──
    ck = torch.load(args.student_init, map_location="cpu")
    meta = ck.get("meta", {})
    model = build_model(meta, device)
    model.load_state_dict(ck["model"], strict=False)
    print(f"已載入初始 student（{meta}）")

    if args.freeze_patch_embed:
        n_frozen = 0
        for p in model.aggregator.patch_embed.parameters():
            p.requires_grad = False
            n_frozen += p.numel()
        print(f"已凍結 patch_embed：{n_frozen/1e6:.1f}M 參數")

    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"可訓練參數：{sum(p.numel() for p in trainable)/1e6:.1f}M")

    loss_fn = DistillLoss().to(device)
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    ds = TeacherDataset(args.data_root, "train", args.img_per_seq,
                        args.steps_per_epoch * args.batch_scenes, gt_pose=bool(args.gt_pose))
    dl = DataLoader(ds, batch_size=args.batch_scenes, shuffle=True,
                    num_workers=args.num_workers, pin_memory=False, drop_last=True)

    total_steps = args.epochs * args.steps_per_epoch
    warmup = int(total_steps * args.warmup_frac)
    step = 0
    best_rot = float("inf")

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        run = {}
        for batch in dl:
            lr = cosine_warmup(step, total_steps, warmup, args.lr)
            for g in opt.param_groups:
                g["lr"] = lr

            batch = {k: v.to(device) for k, v in batch.items()}
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=dtype, enabled=(device == "cuda")):
                pred = model(batch["images"])
                # camera/depth heads 內部已關 autocast；loss 用 fp32
                total, logs = loss_fn(pred, batch)
            if not torch.isfinite(total):
                print(f"[step {step}] non-finite loss，跳過"); step += 1; continue
            total.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()

            # GPU 節流：讓出時間給顯示驅動，避免 100% 佔用導致黑頻
            if args.throttle > 0:
                if device == "cuda":
                    torch.cuda.synchronize()
                time.sleep(args.throttle)

            for k, v in logs.items():
                run[k] = run.get(k, 0.0) + float(v)
            step += 1
            if step % 50 == 0:
                msg = " ".join(f"{k}={run[k]/50:.4f}" for k in ["loss_rot", "loss_transfov", "loss_depth", "objective"] if k in run)
                print(f"[e{epoch} s{step}] lr={lr:.2e} {msg}")
                run = {}

        print(f"epoch {epoch} 完成（{time.time()-t0:.0f}s）")

        # ── periodic eval ──
        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            model.eval()
            m = quick_eval(model, str(Path(args.data_root) / "test"), args.eval_scenes, device, dtype)
            print(f"  [EVAL e{epoch}] rot_err={m['mean_rot_err']:.2f}  rra15={m['rra_15']:.3f}  "
                  f"abs_rel={m['abs_rel']:.4f}  d1={m['delta_1']:.3f}")
            torch.save({"model": model.state_dict(), "meta": meta, "epoch": epoch}, out / "last.pt")
            if m["mean_rot_err"] < best_rot:
                best_rot = m["mean_rot_err"]
                torch.save({"model": model.state_dict(), "meta": meta, "epoch": epoch,
                            "eval": m}, out / "best.pt")
                print(f"  ✓ 新 best（rot_err={best_rot:.2f}），已存 best.pt")

    print(f"\n訓練完成。最佳 rot_err={best_rot:.2f}")


if __name__ == "__main__":
    main()
